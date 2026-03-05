use anyhow::{Context, Result};
use ndarray::{Array1, Array2, Array3};
use ort::{GraphOptimizationLevel, Session, Value};
use tracing::{debug, info};

/// Silero VAD configuration
const SAMPLE_RATE: i64 = 16000;
const FRAME_SIZE: usize = 512; // 32ms at 16kHz
const CONTEXT_SIZE: usize = 64;

/// Voice Activity Detector using Silero VAD ONNX model.
///
/// This runs entirely on CPU and is extremely lightweight (~2ms per frame).
/// It acts as a gate before sending audio to Whisper, saving significant compute.
pub struct VoiceActivityDetector {
    session: Session,
    state: Array3<f32>,        // (2, 1, 128) — internal RNN state
    context: Array2<f32>,      // (1, 64) — audio context window
    config: VadConfig,
    /// State machine for tracking speech boundaries
    speech_state: SpeechState,
}

#[derive(Debug, Clone)]
pub struct VadConfig {
    /// Probability above which we consider speech started
    pub pos_threshold: f32,
    /// Probability below which we consider speech ended
    pub neg_threshold: f32,
    /// Minimum duration of speech to trigger (ms) — prevents short bursts
    pub min_speech_duration_ms: u32,
    /// Minimum silence duration to end speech segment (ms)
    pub min_silence_duration_ms: u32,
    /// Extra audio to keep before speech starts (ms) — captures word onsets
    pub speech_pad_ms: u32,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            pos_threshold: 0.5,
            neg_threshold: 0.35,
            min_speech_duration_ms: 250,
            min_silence_duration_ms: 700,
            speech_pad_ms: 200,
        }
    }
}

/// Tracks the state machine for speech detection with hysteresis
#[derive(Debug)]
struct SpeechState {
    speaking: bool,
    /// Timestamp (in ms) when the last transition occurred
    last_transition_ms: u64,
    /// Whether the *previous* frame was above the positive threshold
    prev_frame_speaking: bool,
    /// Current time counter in ms
    current_ms: u64,
}

impl SpeechState {
    fn new() -> Self {
        Self {
            speaking: false,
            last_transition_ms: 0,
            prev_frame_speaking: false,
            current_ms: 0,
        }
    }

    fn reset(&mut self) {
        *self = Self::new();
    }
}

/// Result of processing a single audio frame through VAD
#[derive(Debug, Clone)]
pub struct VadResult {
    /// Raw speech probability from the model (0.0 - 1.0)
    pub speech_probability: f32,
    /// Whether we are currently in a speech segment (with hysteresis applied)
    pub is_speech: bool,
    /// Whether a speech segment just started this frame
    pub speech_started: bool,
    /// Whether a speech segment just ended this frame
    pub speech_ended: bool,
}

impl VoiceActivityDetector {
    /// Create a new VAD instance from a Silero ONNX model file.
    pub fn new(model_path: &str, config: VadConfig) -> Result<Self> {
        info!("Loading Silero VAD model from: {}", model_path);

        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .commit_from_file(model_path)
            .context("Failed to load Silero VAD ONNX model")?;

        // Initialize state tensors
        let state = Array3::<f32>::zeros((2, 1, 128));
        let context = Array2::<f32>::zeros((1, CONTEXT_SIZE));

        info!("✅ Silero VAD loaded successfully");

        Ok(Self {
            session,
            state,
            context,
            config,
            speech_state: SpeechState::new(),
        })
    }

    /// Process a single audio frame (512 samples at 16kHz = 32ms).
    ///
    /// Returns the VAD result including speech probability and state transitions.
    pub fn process_frame(&mut self, frame: &[f32]) -> Result<VadResult> {
        if frame.len() != FRAME_SIZE {
            anyhow::bail!(
                "VAD frame must be exactly {} samples, got {}",
                FRAME_SIZE,
                frame.len()
            );
        }

        // Save context from end of this frame for next iteration
        let next_context = Array2::from_shape_vec(
            (1, CONTEXT_SIZE),
            frame[FRAME_SIZE - CONTEXT_SIZE..].to_vec(),
        )?;

        // Build input: concatenate [context, frame] → shape (1, CONTEXT_SIZE + FRAME_SIZE)
        let frame_arr = Array2::from_shape_vec((1, FRAME_SIZE), frame.to_vec())?;

        let mut input_data = Vec::with_capacity(CONTEXT_SIZE + FRAME_SIZE);
        input_data.extend_from_slice(self.context.as_slice().unwrap());
        input_data.extend_from_slice(frame_arr.as_slice().unwrap());
        let input = Array2::from_shape_vec((1, CONTEXT_SIZE + FRAME_SIZE), input_data)?;

        // Create ONNX Runtime input values
        let sr_array = Array1::from_vec(vec![SAMPLE_RATE]);

        let inputs = vec![
            Value::from_array(input.view())?,
            Value::from_array(sr_array.view())?,
            Value::from_array(self.state.view())?,
        ];

        // Run inference
        let outputs = self.session.run(ort::inputs![
            "input" => Value::from_array(input.view())?,
            "sr" => Value::from_array(sr_array.view())?,
            "state" => Value::from_array(self.state.view())?,
        ]?)?;

        // Extract speech probability (output[0]) and updated state (output[1])
        let output_tensor = outputs[0].try_extract_tensor::<f32>()?;
        let speech_prob = output_tensor.as_slice().unwrap()[0];

        let new_state = outputs[1].try_extract_tensor::<f32>()?;
        self.state = new_state
            .to_owned()
            .into_shape_with_order((2, 1, 128))?;

        // Update context for next frame
        self.context = next_context;

        // Apply state machine with hysteresis
        let was_speaking = self.speech_state.speaking;
        self.update_speech_state(speech_prob);
        let is_speaking = self.speech_state.speaking;

        Ok(VadResult {
            speech_probability: speech_prob,
            is_speech: is_speaking,
            speech_started: !was_speaking && is_speaking,
            speech_ended: was_speaking && !is_speaking,
        })
    }

    /// Update the speech state machine with hysteresis to avoid flickering.
    fn update_speech_state(&mut self, speech_prob: f32) {
        let frame_ms = (FRAME_SIZE as u64 * 1000) / SAMPLE_RATE as u64; // 32ms
        self.speech_state.current_ms += frame_ms;

        // Determine if this frame is speech based on thresholds with hysteresis
        let curr_frame_speaking = if speech_prob > self.config.pos_threshold {
            true
        } else if speech_prob < self.config.neg_threshold {
            false
        } else {
            // In the hysteresis zone — keep previous state
            self.speech_state.prev_frame_speaking
        };

        // Track transitions
        if curr_frame_speaking != self.speech_state.prev_frame_speaking {
            self.speech_state.last_transition_ms = self.speech_state.current_ms;
        }

        // Apply minimum duration constraints
        if curr_frame_speaking {
            if !self.speech_state.speaking {
                let speech_duration =
                    self.speech_state.current_ms - self.speech_state.last_transition_ms;
                if speech_duration >= self.config.min_speech_duration_ms as u64 {
                    self.speech_state.speaking = true;
                    debug!(
                        "🟢 Speech START (prob={:.3}, duration={}ms)",
                        speech_prob, speech_duration
                    );
                }
            }
        } else if self.speech_state.speaking {
            let silence_duration =
                self.speech_state.current_ms - self.speech_state.last_transition_ms;
            if silence_duration >= self.config.min_silence_duration_ms as u64 {
                self.speech_state.speaking = false;
                debug!(
                    "🔴 Speech END (prob={:.3}, silence={}ms)",
                    speech_prob, silence_duration
                );
            }
        }

        self.speech_state.prev_frame_speaking = curr_frame_speaking;
    }

    /// Reset the VAD state (call between interactions).
    pub fn reset(&mut self) {
        self.state = Array3::<f32>::zeros((2, 1, 128));
        self.context = Array2::<f32>::zeros((1, CONTEXT_SIZE));
        self.speech_state.reset();
    }

    /// Get the required frame size in samples.
    pub fn frame_size(&self) -> usize {
        FRAME_SIZE
    }
}

/// Collects audio frames during a VAD-detected speech segment.
///
/// This struct manages the pre-roll buffer (to capture the beginning of words)
/// and accumulates speech frames until silence is detected.
pub struct VadAudioCollector {
    /// Pre-roll circular buffer — keeps recent frames in case speech starts
    pre_roll_buffer: Vec<Vec<f32>>,
    pre_roll_max_frames: usize,
    pre_roll_write_idx: usize,
    pre_roll_count: usize,

    /// Accumulated speech audio
    speech_audio: Vec<f32>,

    /// Whether we are currently collecting speech
    collecting: bool,
}

impl VadAudioCollector {
    pub fn new(speech_pad_ms: u32) -> Self {
        // Calculate how many frames fit in the speech pad
        let frame_ms = (FRAME_SIZE as u32 * 1000) / SAMPLE_RATE as u32;
        let pre_roll_frames = (speech_pad_ms / frame_ms).max(1) as usize;

        Self {
            pre_roll_buffer: vec![vec![0.0; FRAME_SIZE]; pre_roll_frames],
            pre_roll_max_frames: pre_roll_frames,
            pre_roll_write_idx: 0,
            pre_roll_count: 0,
            speech_audio: Vec::new(),
            collecting: false,
        }
    }

    /// Feed a frame and VAD result. Returns collected speech audio when speech ends.
    pub fn feed(&mut self, frame: &[f32], vad_result: &VadResult) -> Option<Vec<f32>> {
        if vad_result.speech_started {
            self.collecting = true;
            self.speech_audio.clear();

            // Flush pre-roll buffer (the audio just before speech started)
            let start = if self.pre_roll_count < self.pre_roll_max_frames {
                0
            } else {
                self.pre_roll_write_idx
            };
            let count = self.pre_roll_count.min(self.pre_roll_max_frames);
            for i in 0..count {
                let idx = (start + i) % self.pre_roll_max_frames;
                self.speech_audio.extend_from_slice(&self.pre_roll_buffer[idx]);
            }
        }

        if self.collecting {
            self.speech_audio.extend_from_slice(frame);
        }

        // Always update pre-roll buffer
        self.pre_roll_buffer[self.pre_roll_write_idx] = frame.to_vec();
        self.pre_roll_write_idx = (self.pre_roll_write_idx + 1) % self.pre_roll_max_frames;
        self.pre_roll_count += 1;

        if vad_result.speech_ended && self.collecting {
            self.collecting = false;
            let audio = std::mem::take(&mut self.speech_audio);
            self.pre_roll_count = 0;
            self.pre_roll_write_idx = 0;
            return Some(audio);
        }

        None
    }

    /// Check if currently collecting speech
    pub fn is_collecting(&self) -> bool {
        self.collecting
    }

    /// Force-flush any accumulated audio (e.g., on timeout)
    pub fn flush(&mut self) -> Option<Vec<f32>> {
        if self.collecting && !self.speech_audio.is_empty() {
            self.collecting = false;
            let audio = std::mem::take(&mut self.speech_audio);
            self.pre_roll_count = 0;
            self.pre_roll_write_idx = 0;
            Some(audio)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vad_config_defaults() {
        let config = VadConfig::default();
        assert!(config.pos_threshold > config.neg_threshold);
        assert!(config.min_speech_duration_ms > 0);
        assert!(config.min_silence_duration_ms > 0);
    }

    #[test]
    fn test_vad_audio_collector_preroll() {
        let mut collector = VadAudioCollector::new(200);

        // Feed some silence frames
        let silence = vec![0.0f32; FRAME_SIZE];
        let silent_result = VadResult {
            speech_probability: 0.0,
            is_speech: false,
            speech_started: false,
            speech_ended: false,
        };

        for _ in 0..10 {
            assert!(collector.feed(&silence, &silent_result).is_none());
        }

        // Speech starts — pre-roll should be included
        let speech_start = VadResult {
            speech_probability: 0.9,
            is_speech: true,
            speech_started: true,
            speech_ended: false,
        };
        let speech_frame = vec![0.5f32; FRAME_SIZE];
        assert!(collector.feed(&speech_frame, &speech_start).is_none());
        assert!(collector.is_collecting());

        // Speech ends
        let speech_end = VadResult {
            speech_probability: 0.1,
            is_speech: false,
            speech_started: false,
            speech_ended: true,
        };
        let result = collector.feed(&silence, &speech_end);
        assert!(result.is_some());

        let audio = result.unwrap();
        // Should contain pre-roll + speech frames + end frame
        assert!(audio.len() > FRAME_SIZE);
    }
}