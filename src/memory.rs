use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use tracing::{debug, info};
use uuid::Uuid;

/// A single exchange in the conversation (user message + Jarvis response).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationTurn {
    pub id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub user_message: String,
    pub assistant_response: String,
    /// What sensor was queried (if any)
    pub sensor_context: Option<SensorContext>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorContext {
    pub sensor_type: String,
    pub location: Option<String>,
    pub value: serde_json::Value,
}

/// Configuration for conversation memory behavior.
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    /// Maximum number of conversation turns to remember
    pub max_turns: usize,
    /// Maximum age of memories before they're dropped (in seconds)
    pub max_age_secs: u64,
    /// Maximum total token estimate for context window management
    /// (rough estimate: 1 token ≈ 4 chars)
    pub max_context_chars: usize,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            max_turns: 20,
            max_age_secs: 3600, // 1 hour
            max_context_chars: 4000, // ~1000 tokens — safe for 7B models
        }
    }
}

/// In-memory conversation store with automatic pruning.
///
/// Maintains a rolling window of recent conversations that gets
/// serialized into the LLM prompt to provide context.
pub struct ConversationMemory {
    turns: VecDeque<ConversationTurn>,
    config: MemoryConfig,
    session_id: Uuid,
}

impl ConversationMemory {
    pub fn new(config: MemoryConfig) -> Self {
        let session_id = Uuid::new_v4();
        info!(
            "📝 Conversation memory initialized (session: {}, max_turns: {}, max_age: {}s)",
            session_id, config.max_turns, config.max_age_secs
        );
        Self {
            turns: VecDeque::new(),
            config,
            session_id,
        }
    }

    /// Record a completed conversation turn.
    pub fn add_turn(
        &mut self,
        user_message: &str,
        assistant_response: &str,
        sensor_context: Option<SensorContext>,
    ) {
        let turn = ConversationTurn {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            user_message: user_message.to_string(),
            assistant_response: assistant_response.to_string(),
            sensor_context,
        };

        debug!(
            "📝 Adding turn #{}: user='{}' → assistant='{}'",
            self.turns.len() + 1,
            truncate_str(user_message, 50),
            truncate_str(assistant_response, 50),
        );

        self.turns.push_back(turn);
        self.prune();
    }

    /// Generate a formatted conversation history string for inclusion in LLM prompts.
    ///
    /// Returns recent conversation turns formatted as a dialogue, respecting
    /// the max context size to avoid overloading the LLM's context window.
    pub fn format_for_prompt(&self) -> String {
        if self.turns.is_empty() {
            return String::new();
        }

        let mut result = String::from("## Recent Conversation History\n");
        let mut total_chars = result.len();
        let mut included_turns = Vec::new();

        // Build from most recent backwards, then reverse
        for turn in self.turns.iter().rev() {
            let turn_text = format!(
                "User: {}\nJarvis: {}\n",
                turn.user_message, turn.assistant_response
            );

            if total_chars + turn_text.len() > self.config.max_context_chars {
                break;
            }

            total_chars += turn_text.len();
            included_turns.push(turn_text);
        }

        // Reverse to get chronological order
        included_turns.reverse();

        for turn_text in included_turns {
            result.push_str(&turn_text);
            result.push('\n');
        }

        result
    }

    /// Generate a summary of sensor values mentioned in recent history.
    ///
    /// This helps the LLM understand trends, e.g., "the temperature was 22°C
    /// five minutes ago and now it's 25°C".
    pub fn format_sensor_summary(&self) -> String {
        let sensor_turns: Vec<&ConversationTurn> = self
            .turns
            .iter()
            .filter(|t| t.sensor_context.is_some())
            .collect();

        if sensor_turns.is_empty() {
            return String::new();
        }

        let mut summary = String::from("## Previous Sensor Readings\n");
        // Only include the last 5 sensor readings to keep it concise
        let recent = &sensor_turns[sensor_turns.len().saturating_sub(5)..];

        for turn in recent {
            if let Some(ctx) = &turn.sensor_context {
                let ago = Utc::now()
                    .signed_duration_since(turn.timestamp)
                    .num_seconds();
                let ago_str = format_duration_ago(ago);

                summary.push_str(&format!(
                    "- {} {} ago: {}\n",
                    ctx.sensor_type,
                    ago_str,
                    ctx.value
                ));
            }
        }

        summary
    }

    /// Get the number of stored turns.
    pub fn len(&self) -> usize {
        self.turns.len()
    }

    /// Check if memory is empty.
    pub fn is_empty(&self) -> bool {
        self.turns.is_empty()
    }

    /// Get the session ID.
    pub fn session_id(&self) -> Uuid {
        self.session_id
    }

    /// Clear all conversation history (e.g., on "forget everything").
    pub fn clear(&mut self) {
        info!("🗑️ Conversation memory cleared");
        self.turns.clear();
    }

    /// Prune old turns based on max_turns and max_age.
    fn prune(&mut self) {
        // Remove excess turns
        while self.turns.len() > self.config.max_turns {
            self.turns.pop_front();
        }

        // Remove expired turns
        let cutoff = Utc::now()
            - chrono::Duration::seconds(self.config.max_age_secs as i64);

        while let Some(front) = self.turns.front() {
            if front.timestamp < cutoff {
                debug!("🗑️ Pruning expired turn from {}", front.timestamp);
                self.turns.pop_front();
            } else {
                break; // Turns are in chronological order
            }
        }
    }
}

/// Format a duration in seconds to a human-readable string.
fn format_duration_ago(seconds: i64) -> String {
    if seconds < 60 {
        format!("{}s", seconds)
    } else if seconds < 3600 {
        format!("{}m", seconds / 60)
    } else {
        format!("{}h {}m", seconds / 3600, (seconds % 3600) / 60)
    }
}

/// Truncate a string for debug logging.
fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_format_turns() {
        let mut memory = ConversationMemory::new(MemoryConfig {
            max_turns: 5,
            max_age_secs: 3600,
            max_context_chars: 4000,
        });

        memory.add_turn(
            "What's the temperature in the living room?",
            "The living room temperature is currently 22.5°C.",
            Some(SensorContext {
                sensor_type: "temperature".to_string(),
                location: Some("living_room".to_string()),
                value: serde_json::json!({"value": 22.5, "unit": "°C"}),
            }),
        );

        memory.add_turn(
            "And the humidity?",
            "The humidity in the living room is 45%.",
            Some(SensorContext {
                sensor_type: "humidity".to_string(),
                location: Some("living_room".to_string()),
                value: serde_json::json!({"value": 45, "unit": "%"}),
            }),
        );

        let prompt = memory.format_for_prompt();
        assert!(prompt.contains("temperature"));
        assert!(prompt.contains("humidity"));
        assert_eq!(memory.len(), 2);
    }

    #[test]
    fn test_max_turns_pruning() {
        let mut memory = ConversationMemory::new(MemoryConfig {
            max_turns: 3,
            max_age_secs: 3600,
            max_context_chars: 4000,
        });

        for i in 0..5 {
            memory.add_turn(
                &format!("Question {}", i),
                &format!("Answer {}", i),
                None,
            );
        }

        assert_eq!(memory.len(), 3);
        // Should keep only the last 3
        let prompt = memory.format_for_prompt();
        assert!(!prompt.contains("Question 0"));
        assert!(!prompt.contains("Question 1"));
        assert!(prompt.contains("Question 2"));
        assert!(prompt.contains("Question 3"));
        assert!(prompt.contains("Question 4"));
    }

    #[test]
    fn test_context_char_limit() {
        let mut memory = ConversationMemory::new(MemoryConfig {
            max_turns: 100,
            max_age_secs: 3600,
            max_context_chars: 200, // very small
        });

        for i in 0..20 {
            memory.add_turn(
                &format!("This is a longer question number {}", i),
                &format!("This is a longer answer number {}", i),
                None,
            );
        }

        let prompt = memory.format_for_prompt();
        // Should be under the limit
        assert!(prompt.len() <= 250); // some slack for header
    }

    #[test]
    fn test_sensor_summary() {
        let mut memory = ConversationMemory::new(MemoryConfig::default());

        memory.add_turn(
            "Temperature?",
            "22°C",
            Some(SensorContext {
                sensor_type: "temperature".to_string(),
                location: Some("living_room".to_string()),
                value: serde_json::json!(22.5),
            }),
        );

        let summary = memory.format_sensor_summary();
        assert!(summary.contains("temperature"));
    }

    #[test]
    fn test_clear() {
        let mut memory = ConversationMemory::new(MemoryConfig::default());
        memory.add_turn("Hello", "Hi!", None);
        assert!(!memory.is_empty());
        memory.clear();
        assert!(memory.is_empty());
    }
}