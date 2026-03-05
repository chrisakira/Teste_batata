/// Check if the transcribed text contains the wake word "Hello Jarvis"
/// Uses fuzzy matching to account for transcription inaccuracies.
pub fn is_wake_word(transcript: &str) -> bool {
    let lower = transcript.to_lowercase();
    let lower = lower.trim();

    // Direct matches
    if lower.contains("hello jarvis")
        || lower.contains("hey jarvis")
        || lower.contains("hi jarvis")
        || lower.contains("ok jarvis")
    {
        return true;
    }

    // Fuzzy matching: check if words are close enough
    // Whisper sometimes transcribes "Jarvis" as "Jarves", "Jarvas", etc.
    let words: Vec<&str> = lower.split_whitespace().collect();
    for window in words.windows(2) {
        let greet = window[0];
        let name = window[1];

        let is_greeting = matches!(greet, "hello" | "hey" | "hi" | "ok" | "yo");
        let is_jarvis = fuzzy_match_jarvis(name);

        if is_greeting && is_jarvis {
            return true;
        }
    }

    false
}

fn fuzzy_match_jarvis(word: &str) -> bool {
    if word.len() < 4 || word.len() > 8 {
        return false;
    }
    // Simple edit distance check — must start with "jar" or "jer"
    (word.starts_with("jar") || word.starts_with("jer"))
        && (word.ends_with("vis") || word.ends_with("ves") || word.ends_with("vas"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wake_word_detection() {
        assert!(is_wake_word("Hello Jarvis"));
        assert!(is_wake_word("hello jarvis"));
        assert!(is_wake_word("Hey Jarvis, what's up?"));
        assert!(is_wake_word("Hi Jarvis"));
        assert!(!is_wake_word("What's the temperature?"));
        assert!(!is_wake_word("Hello there"));
    }
}