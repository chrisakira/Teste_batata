use anyhow::{Context, Result};
use std::sync::{Arc, Mutex};
use tracing::{debug, info, warn};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

/// Streaming STT configuration
#[derive(Debug, Clone)]
pub struct SttConfig {
    /// How often to run inference (ms)
    pub step_ms: u32,
    /// Total audio window to analyze (ms)
    pub length_ms: u32,
    /// Overlap to keep between windows (ms) — prevents cutting words
    pub keep_ms: u32,
    /// Number of threads for Whisper inference
    pub n_threads: i32,
    /// Language code
    pub language: String,
}

impl Default for SttConfig {
    fn default() -> Self {
        Self {
            step_ms: 3000,     // Run inference every 3s
            length_ms: 10000,  // Analyze up to 10s window
            keep_ms: 200,      // 200ms overlap between windows
            n_threads: 4,
            language: "en".to_string(),
        }
    }
}

/// Speech-to-Text engine with streaming support.
///
/// Instead of capturing a fixed duration and transcribing, this engine
/// maintains a rolling audio buffer and processes overlapping windows.
/// This provides:
/// - Lower latency (don't wait for a fixed window to fill)
/// - Better accuracy (overlapping windows prevent word boundary issues)
/// - Natural integration with VAD (process only speech segments)
pub struct SttEngine {
    ctx: WhisperContext,
    config: SttConfig,
}

impl SttEngine {
    pub fn new(model_path: &str, config: SttConfig) -> Result<Self> {
        info!("Loading Whisper model from: {}", model_path);
        let ctx = WhisperContext::new_with_params(
            model_path,
            WhisperContextParameters::default(),
        )
        .map_err(|e| anyhow::anyhow!("Failed to load Whisper model: {:?}", e))?;

        info!(
            "✅ Whisper model loaded (step={}ms, window={}ms, keep={}ms)",
            config.step_ms, config.length_ms, config.keep_ms
        );

        Ok(Self { ctx, config })
    }

    /// Transcribe a complete audio buffer (used for VAD-segmented speech).
    ///
    /// This is the simplest mode: VAD collects a complete speech segment,
    /// then we transcribe the whole thing at once.
    pub fn transcribe(&self, audio_data: &[f32]) -> Result<String> {
        if audio_data.is_empty() {
            return Ok(String::new());
        }

        let mut params = self.build_params();
        params.set_single_segment(false); // Allow multiple segments for longer speech

        let mut state = self
            .ctx
            .create_state()
            .map_err(|e| anyhow::anyhow!("Failed to create whisper state: {:?}", e))?;

        state
            .full(params, audio_data)
            .map_err(|e| anyhow::anyhow!("Whisper inference failed: {:?}", e))?;

        let num_segments = state
            .full_n_segments()
            .map_err(|e| anyhow::anyhow!("Failed to get segments: {:?}", e))?;

        let mut text = String::new();
        for i in 0..num_segments {
            if let Ok(segment) = state.full_get_segment_text(i) {
                text.push_str(&segment);
                text.push(' ');
            }
        }

        let result = text.trim().to_string();
        if !result.is_empty() {
            debug!("STT result: {}", result);
        }
        Ok(result)
    }

    /// Create a streaming transcription session.
    ///
    /// Returns a `StreamingSession` that can be fed audio chunks incrementally.
    /// Useful for real-time wake-word detection where you want to continuously
    /// process microphone input without blocking.
    pub fn create_streaming_session(&self) -> Result<StreamingSession> {
        let state = self
            .ctx
            .create_state()
            .map_err(|e| anyhow::anyhow!("Failed to create whisper state: {:?}", e))?;

        Ok(StreamingSession {
            state,
            config: self.config.clone(),
            audio_buffer: Vec::new(),
            previous_tail: Vec::new(),
            params_builder: StreamParamsBuilder {
                n_threads: self.config.n_threads,
                language: self.config.language.clone(),
            },
        })
    }

    fn build_params(&self) -> FullParams<'static, 'static> {
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });

        params.set_n_threads(self.config.n_threads);
        params.set_language(Some("en"));
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_suppress_blank(true);
        params.set_no_context(true);
        params.set_single_segment(true);
        // Suppress non-speech tokens for cleaner output
        params.set_suppress_non_speech_tokens(true);

        params
    }
}

/// Helper struct for deferred parameter building (avoids lifetime issues).
struct StreamParamsBuilder {
    n_threads: i32,
    language: String,
}

impl StreamParamsBuilder {
    fn build(&self) -> FullParams<'static, 'static> {
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });

        params.set_n_threads(self.n_threads);
        params.set_language(Some("en"));
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_suppress_blank(true);
        params.set_no_context(true);
        params.set_single_segment(true);
        params.set_suppress_non_speech_tokens(true);

        params
    }
}

/// A streaming transcription session that processes audio incrementally.
///
/// This implements the rolling-window approach from whisper.cpp's stream example:
///
/// ```text
/// Time ─────────────────────────────────────────────▶
///
/// Window 1: [==========]
///                  keep ──┐
/// Window 2:        [=====|=====]
///                             keep ──┐
/// Window 3:                   [=====|=====]
/// ```
///
/// Each window overlaps with the previous one by `keep_ms` to avoid
/// cutting words at boundaries. Only the NEW text from each window
/// is returned, preventing duplication.
pub struct StreamingSession<'a> {
    state: whisper_rs::WhisperState<'a>,
    config: SttConfig,
    /// Rolling audio buffer
    audio_buffer: Vec<f32>,
    /// Tail of previous window (overlap region)
    previous_tail: Vec<f32>,
    params_builder: StreamParamsBuilder,
}

impl<'a> StreamingSession<'a> {
    /// Feed new audio samples into the streaming session.
    ///
    /// Returns `Some(text)` when enough audio has accumulated for a
    /// transcription step. Returns `None` if more audio is needed.
    pub fn feed_audio(&mut self, samples: &[f32]) -> Result<Option<String>> {
        self.audio_buffer.extend_from_slice(samples);

        let step_samples = (self.config.step_ms as usize * 16000) / 1000;
        let max_samples = (self.config.length_ms as usize * 16000) / 1000;
        let keep_samples = (self.config.keep_ms as usize * 16000) / 1000;

        // Not enough new audio for a transcription step yet
        if self.audio_buffer.len() < step_samples {
            return Ok(None);
        }

        // Build the analysis window: previous_tail + new audio
        let mut window = Vec::with_capacity(max_samples);
        window.extend_from_slice(&self.previous_tail);
        window.extend_from_slice(&self.audio_buffer);

        // Truncate to max window length (keep the most recent audio)
        if window.len() > max_samples {
            let excess = window.len() - max_samples;
            window.drain(..excess);
        }

        // Save the tail for the next window's overlap
        if window.len() > keep_samples {
            self.previous_tail = window[window.len() - keep_samples..].to_vec();
        } else {
            self.previous_tail = window.clone();
        }

        // Clear the buffer (we've consumed it)
        self.audio_buffer.clear();

        // Run Whisper inference on the window
        let params = self.params_builder.build();

        match self.state.full(params, &window) {
            Ok(()) => {}
            Err(e) => {
                warn!("Whisper streaming inference error: {:?}", e);
                return Ok(None);
            }
        }

        // Extract text from all segments
        let num_segments = self
            .state
            .full_n_segments()
            .map_err(|e| anyhow::anyhow!("Failed to get segments: {:?}", e))?;

        let mut text = String::new();
        for i in 0..num_segments {
            if let Ok(segment) = self.state.full_get_segment_text(i) {
                text.push_str(&segment);
                text.push(' ');
            }
        }

        let result = text.trim().to_string();
        if result.is_empty() {
            Ok(None)
        } else {
            debug!("Streaming STT: {}", result);
            Ok(Some(result))
        }
    }

    /// Finalize the session and transcribe any remaining audio.
    pub fn finalize(&mut self) -> Result<Option<String>> {
        if self.audio_buffer.is_empty() && self.previous_tail.is_empty() {
            return Ok(None);
        }

        let mut final_audio = Vec::new();
        final_audio.extend_from_slice(&self.previous_tail);
        final_audio.extend_from_slice(&self.audio_buffer);

        if final_audio.is_empty() {
            return Ok(None);
        }

        let params = self.params_builder.build();

        self.state
            .full(params, &final_audio)
            .map_err(|e| anyhow::anyhow!("Whisper finalize failed: {:?}", e))?;

        let num_segments = self
            .state
            .full_n_segments()
            .map_err(|e| anyhow::anyhow!("Failed to get segments: {:?}", e))?;

        let mut text = String::new();
        for i in 0..num_segments {
            if let Ok(segment) = self.state.full_get_segment_text(i) {
                text.push_str(&segment);
                text.push(' ');
            }
        }

        self.audio_buffer.clear();
        self.previous_tail.clear();

        let result = text.trim().to_string();
        if result.is_empty() {
            Ok(None)
        } else {
            Ok(Some(result))
        }
    }

    /// Reset the session for a new utterance.
    pub fn reset(&mut self) {
        self.audio_buffer.clear();
        self.previous_tail.clear();
    }
}