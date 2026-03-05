mod audio;
mod llm;
mod memory;
mod sensors;
mod stt;
mod tts;
mod vad;
mod wake_word;

use memory::{ConversationMemory, MemoryConfig, SensorContext};
use stt::SttConfig;
use vad::{VadAudioCollector, VadConfig, VoiceActivityDetector};

use std::time::{Duration, Instant};
use tracing::{debug, error, info, warn};

/// Maximum time to wait for a command after wake word (seconds)
const COMMAND_TIMEOUT_SECS: u64 = 15;
/// Maximum speech duration before forced cutoff (seconds)
const MAX_SPEECH_DURATION_SECS: u64 = 30;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    info!("🤖 Jarvis Assistant v0.2 starting up...");
    info!("   Enhancements: VAD + Streaming STT + Conversation Memory");

    // ── Configuration ────────────────────────────────────────────────────
    let home = std::env::var("HOME").unwrap_or_else(|_| "/home/user".to_string());

    let whisper_model_path = std::env::var("WHISPER_MODEL_PATH")
        .unwrap_or_else(|_| format!("{}/jarvis/models/ggml-base.en.bin", home));
    let vad_model_path = std::env::var("VAD_MODEL_PATH")
        .unwrap_or_else(|_| format!("{}/jarvis/models/silero_vad.onnx", home));
    let piper_model_path = std::env::var("PIPER_MODEL_PATH")
        .unwrap_or_else(|_| format!("{}/jarvis/voices/en_US-lessac-medium.onnx", home));
    let ollama_url =
        std::env::var("OLLAMA_URL").unwrap_or_else(|_| "http://localhost:11434".to_string());
    let ollama_model = std::env::var("OLLAMA_MODEL")
        .unwrap_or_else(|_| "mistral:7b-instruct-v0.3-q5_K_M".to_string());
    let sensor_api_url =
        std::env::var("SENSOR_API_URL").unwrap_or_else(|_| "http://localhost:8080".to_string());

    // ── Initialize Components ────────────────────────────────────────────
    let stt_engine = stt::SttEngine::new(&whisper_model_path, SttConfig::default())?;
    let mut vad_detector = VoiceActivityDetector::new(&vad_model_path, VadConfig::default())?;
    let tts_engine = tts::TtsEngine::new(&piper_model_path)?;
    let llm_client = llm::LlmClient::new(&ollama_url, &ollama_model);
    let sensor_client = sensors::SensorClient::new(&sensor_api_url);
    let mut memory = ConversationMemory::new(MemoryConfig::default());

    // ── Start Audio Capture ──────────────────────────────────────────────
    let audio_capture = audio::AudioCapture::start()?;

    info!("✅ All components initialized");
    info!("🎤 Listening for 'Hello Jarvis' (with VAD)...");

    // ── Main Loop ────────────────────────────────────────────────────────
    let frame_size = vad_detector.frame_size(); // 512 samples = 32ms
    let mut wake_collector = VadAudioCollector::new(200);
    let mut streaming_session = stt_engine.create_streaming_session()?;

    loop {
        // Wait for enough samples for one VAD frame
        while audio_capture.available_samples() < frame_size {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        let samples = audio_capture.read_samples(frame_size);
        if samples.len() < frame_size {
            continue;
        }

        // ── Phase 1: VAD-gated Wake Word Detection ───────────────────
        let vad_result = vad_detector.process_frame(&samples)?;

        // Feed audio to the collector; it handles pre-roll and accumulation
        if let Some(speech_audio) = wake_collector.feed(&samples, &vad_result) {
            // Complete speech segment detected — check for wake word
            let transcript = stt_engine.transcribe(&speech_audio)?;
            debug!("Wake word check: '{}'", transcript);

            if wake_word::is_wake_word(&transcript) {
                info!("🔔 Wake word detected! Entering command mode...");
                vad_detector.reset();

                // Acknowledge
                tts_engine.speak("Yes?").await?;

                // ── Phase 2: VAD-gated Command Capture with Streaming STT ──
                let command_text =
                    capture_command(&audio_capture, &mut vad_detector, &stt_engine).await?;

                if command_text.is_empty() {
                    tts_engine
                        .speak("I didn't catch that. Please try again.")
                        .await?;
                    vad_detector.reset();
                    streaming_session.reset();
                    continue;
                }

                info!("📝 Command: {}", command_text);

                // ── Phase 3: Intent Detection (with memory context) ──────
                let intent = llm_client.detect_intent(&command_text, &memory).await?;
                info!("🎯 Intent: {:?}", intent);

                // Handle special intents
                if intent.action == "clear_memory" {
                    memory.clear();
                    let response = "Very well. I've cleared my memory of our previous conversations. Fresh start.";
                    