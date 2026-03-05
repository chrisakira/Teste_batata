use reqwest::Client;
use serde_json::Value;
use tracing::info;

pub struct SensorClient {
    client: Client,
    base_url: String,
}

impl SensorClient {
    pub fn new(base_url: &str) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.to_string(),
        }
    }

    /// Fetches sensor data from your local API.
    /// Adapt the endpoint structure to match YOUR actual sensor API.
    pub async fn get_sensor_data(&self, sensor_type: &str) -> anyhow::Result<Value> {
        // Adjust this URL pattern to match your actual sensor API
        // Examples:
        //   GET /api/sensors/temperature
        //   GET /api/sensors/humidity?location=living_room
        let url = format!("{}/api/sensors/{}", self.base_url, sensor_type);
        info!("Fetching sensor data from: {}", url);

        let response = self.client.get(&url).send().await?;

        if !response.status().is_success() {
            anyhow::bail!(
                "Sensor API returned status: {}",
                response.status()
            );
        }

        let data: Value = response.json().await?;
        info!("Sensor data received: {}", data);
        Ok(data)
    }
}