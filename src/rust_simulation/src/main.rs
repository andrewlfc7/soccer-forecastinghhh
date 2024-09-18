use tokio;
use rust_simulation::api_client::{get_league_fixtures, get_match_details};
use rust_simulation::models::{LeagueResponse, MatchDetail};
use std::env;

#[tokio::main]
async fn main() {
    // Example usage
    let league_id = 47; // Replace with actual league ID

    match get_league_fixtures(league_id).await {
        Ok(league_response) => {
            for fixture in league_response.league_overview_matches {
                println!("Fixture ID: {}, Round: {}", fixture.id, fixture.round);

                match get_match_details(fixture.id).await {
                    Ok(match_detail) => {
                        let home_team = match_detail.general.home_team;
                        let away_team = match_detail.general.away_team;
                        println!(
                            "Match ID: {}, Round: {}, Home Team: {} (ID: {}), Away Team: {} (ID: {})",
                            match_detail.general.match_id,
                            match_detail.general.match_round,
                            home_team.name,
                            home_team.id,
                            away_team.name,
                            away_team.id
                        );
                    }
                    Err(e) => eprintln!("Error fetching match details: {}", e),
                }
            }
        }
        Err(e) => eprintln!("Error fetching league fixtures: {}", e),
    }
}
