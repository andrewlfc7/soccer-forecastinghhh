use serde::Deserialize;

#[derive(Deserialize, Debug)]
pub struct Fixture {
    pub id: u32,
    pub round: u32,
}

#[derive(Deserialize, Debug)]
pub struct LeagueResponse {
    pub league_overview_matches: Vec<Fixture>,
}

#[derive(Deserialize, Debug)]
pub struct MatchDetail {
    pub general: GeneralInfo,
}

#[derive(Deserialize, Debug)]
pub struct GeneralInfo {
    pub home_team: TeamInfo,
    pub away_team: TeamInfo,
    pub match_id: u32,
    pub match_round: u32,
}

#[derive(Deserialize, Debug)]
pub struct TeamInfo {
    pub id: u32,
    pub name: String,
}
