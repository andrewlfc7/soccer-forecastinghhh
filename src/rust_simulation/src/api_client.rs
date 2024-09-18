use reqwest::Client;
use crate::models::{LeagueResponse, MatchDetail};

const BASE_URL: &str = "https://www.fotmob.com/api";

pub async fn get_league_fixtures(league_id: u32) -> Result<LeagueResponse, Box<dyn std::error::Error>> {
    let url = format!("{}/leagues?id={}&ccode3=USA_MA", BASE_URL, league_id);
    let client = Client::new();
    let response = client.get(&url).send().await?;

    // Debuging
    let body = response.text().await?;
    println!("Response body: {}", body);

    let league_response: LeagueResponse = serde_json::from_str(&body)?;
    Ok(league_response)
}


pub async fn get_match_details(match_id: u32) -> Result<MatchDetail, Box<dyn std::error::Error>> {
    let url = format!("{}/matchDetails?matchId={}&ccode3=USA&timezone=America%2FChicago&refresh=true&includeBuzzTab=false&acceptLanguage=en-US", BASE_URL, match_id);
    let client = Client::new();
    let response = client.get(&url).send().await?.json::<MatchDetail>().await?;
    Ok(response)
}
