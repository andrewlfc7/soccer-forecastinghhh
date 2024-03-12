import requests
import json
import os


def are_all_matches_finished(season_length, match_round, current_round):
    match_ids = []

    # Collect match IDs for the round before the current round
    for i in range(season_length):
        if match_round[i]['round'] == current_round - 1:
            match_ids.append(match_round[i]['id'])

    # Check if any matches exist for the round before the current round
    if not match_ids:
        return False  # No matches for the round before the current round

    # Check if all matches in the round before the current round are finished
    for match_id in match_ids:
        response = requests.get(f'https://www.fotmob.com/api/matchDetails?matchId={match_id}&ccode3=USA&timezone=America%2FChicago&refresh=true&includeBuzzTab=false&acceptLanguage=en-US')
        data = response.content
        data = json.loads(data)

        # Check if the match data is valid
        if 'general' not in data:
            return False  # Invalid match data

        # Check if the match is finished
        if data['general'].get('finished') == 'true':
            return False  # At least one match is not finished

    return True


def is_model_built(league_name):
    model_path = f'model/model{league_name}.pkl'
    return os.path.exists(model_path)


def get_remaining_matches(shots_data, all_fixtures_df):
    # Step 1: Get unique match IDs from shots_data
    unique_shots_matches = set(shots_data['matchId'].unique())

    # Step 2: Get unique match IDs from all_fixtures_df
    unique_fixtures_matches = set(all_fixtures_df['matchId'].unique())

    # Step 3: Find the difference to get the matches that are not played
    unplayed_matches = unique_fixtures_matches - unique_shots_matches

    # Step 4: Filter all_fixtures_df for the remaining matches
    remaining_matches_df = all_fixtures_df[all_fixtures_df['matchId'].isin(unplayed_matches)]

    # Assuming you want to convert 'matchId' to int, as mentioned in Step 4
    remaining_matches_df['matchId'] = remaining_matches_df['matchId'].astype(int)

    return remaining_matches_df


# def calculate_mean_played_results(played_result):
#     # Group by 'matchId' and calculate the mean, including team names
#     result = played_result.groupby('matchId').agg({
#         'home_goals': 'mean',
#         'away_goals': 'mean',
#         'home_prob': 'mean',
#         'away_prob': 'mean',
#         'draw_prob': 'mean',
#         'home_team_name': 'first',
#         'away_team_name': 'first',
#     }).round(0).reset_index()
#
#     return result
#
# def calculate_mean_unplayed_results(unplayed_result):
#     # Group by 'matchId' and calculate the mean, including team names
#     result = unplayed_result.groupby('matchId').agg({
#         'home_goals': 'mean',
#         'away_goals': 'mean',
#         'home_team_name': 'first',
#         'away_team_name': 'first',
#     }).round(0).reset_index()
#
#     return result
def remove_aborted_matches(df):
    # Filter rows where status is "aborted"
    aborted_matches_index = df[df['status.reason.short'] == 'Ab'].index

    # Drop the rows corresponding to aborted matches
    df.drop(aborted_matches_index, inplace=True)

    # Reset the index after dropping rows
    df.reset_index(drop=True, inplace=True)

    return df
