use crate::memory::{ConversationMemory, SensorContext};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::info;

pub struct LlmClient {
    client: Client,
    base_url: String,
    model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Intent {
    pub action: String,
    pub sensor_type: Option<String>,
    pub location: Option<String>,
    pub raw_query: String,
}

#[derive(Deserialize)]
struct OllamaResponse {
    response: String,
}

impl LlmClient {
    pub fn new(base_url: &str, model: &str) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.to_string(),
            model: model.to_string(),
        }
    }

    /// Detects the user's intent, now with conversation context for pronoun resolution.
    ///
    /// E.g., if the user previously asked about the living room temperature,
    /// and now says "what about the bedroom?", the memory helps resolve intent.
    pub async fn detect_intent(
        &self,
        user_input: &str,
        memory: &ConversationMemory,
    ) -> anyhow::Result<Intent> {
        let memory_context = if !memory.is_empty() {
            format!(
                "\n{}\nUse the conversation history to resolve pronouns and implicit references.\n",
                memory.format_for_prompt()
            )
        } else {
            String::new()
        };

        let prompt = format!(
            r#"You are an intent detection system for a smart home assistant called Jarvis.
Analyze the user's request and respond ONLY with valid JSON (no extra text).

Available sensor types: temperature, humidity, motion, light, door, window, smoke, co2, energy, water
Available locations: living_room, bedroom, kitchen, bathroom, garage, garden, office, hallway
{memory_context}
User said: "{user_input}"

Respond with JSON in this exact format:
{{"action": "query_sensor" or "general_chat", "sensor_type": "type or null", "location": "location or null"}}

Examples:
- "What's the temperature in the living room?" -> {{"action": "query_sensor", "sensor_type": "temperature", "location": "living_room"}}
- "Is there motion in the garage?" -> {{"action": "query_sensor", "sensor_type": "motion", "location": "garage"}}
- "What about the bedroom?" (after asking about temperature) -> {{"action": "query_sensor", "sensor_type": "temperature", "location": "bedroom"}}
- "How are you doing?" -> {{"action": "general_chat", "sensor_type": null, "location": null}}
- "Forget everything" -> {{"action": "clear_memory", "sensor_type": null, "location": null}}

JSON response:"#,
        );

        let response = self.call_ollama(&prompt).await?;
        info!("Intent detection raw response: {}", response);

        let parsed: serde_json::Value = serde_json::from_str(
            response
                .trim()
                .trim_start_matches("```json")
                .trim_end_matches("```")
                .trim(),
        )
        .unwrap_or_else(|_| {
            serde_json::json!({
                "action": "general_chat",
                "sensor_type": null,
                "location": null
            })
        });

        Ok(Intent {
            action: parsed["action"]
                .as_str()
                .unwrap_or("general_chat")
                .to_string(),
            sensor_type: parsed["sensor_type"].as_str().map(|s| s.to_string()),
            location: parsed["location"].as_str().map(|s| s.to_string()),
            raw_query: user_input.to_string(),
        })
    }

    /// Generates a natural language response, enriched with conversation memory.
    pub async fn generate_response(
        &self,
        user_input: &str,
        intent: &Intent,
        sensor_data: Option<&serde_json::Value>,
        memory: &ConversationMemory,
    ) -> anyhow::Result<String> {
        let memory_context = memory.format_for_prompt();
        let sensor_history = memory.format_sensor_summary();

        let prompt = match (&intent.action[..], sensor_data) {
            ("query_sensor", Some(data)) => {
                format!(
                    r#"You are Jarvis, a helpful and friendly smart home assistant.
You speak in a professional but warm tone, similar to the AI assistant from Iron Man.
Keep your responses concise — 1-2 sentences maximum.

{memory_context}
{sensor_history}

The user asked: "{user_input}"

Here is the current sensor data from the home system:
{data}

Provide a natural, conversational response with the sensor information.
Include the actual values and units. Be helpful and proactive if values seem unusual.
If there are previous readings in the history, mention the trend (e.g., "it has risen by 2 degrees since you last asked")."#,
                    data = serde_json::to_string_pretty(data)?
                )
            }
            ("query_sensor", None) => {
                format!(
                    r#"You are Jarvis, a helpful smart home assistant.
{memory_context}
The user asked: "{user_input}"
Unfortunately, the sensor data could not be retrieved.
Apologize briefly and suggest they check the sensor connection."#,
                )
            }
            ("clear_memory", _) => {
                return Ok(
                    "Very well, sir. I've cleared my memory of our previous conversations. Fresh start."
                        .to_string(),
                );
            }
            _ => {
                format!(
                    r#"You are Jarvis, a helpful and witty smart home assistant.
You speak like the AI from Iron Man — professional, warm, with occasional dry humor.
Keep responses concise — 1-2 sentences.

{memory_context}

User said: "{user_input}"

Respond naturally:"#,
                )
            }
        };

        let response = self.call_ollama(&prompt).await?;
        Ok(response.trim().to_string())
    }

    async fn call_ollama(&self, prompt: &str) -> anyhow::Result<String> {
        let url = format!("{}/api/generate", self.base_url);

        let body = serde_json::json!({
            "model": self.model,
            "prompt": prompt,
            "stream": false,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": 256,
            }
        });

        let response = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await?
            .json::<OllamaResponse>()
            .await?;

        Ok(response.response)
    }
}