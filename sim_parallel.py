from pipeline.fotmob import *
from pipeline.model import *
from pipeline.visuals import *
from pipeline.sim import *
from pipeline.utils import *
from joblib import Parallel, delayed

import joblib
from utils import *
import pandas as pd
from google.cloud import storage
from io import BytesIO
import datetime


def main():


    parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    parent_mount_path = '/soccer-forecasting/'

    secret_relative_path = 'forecasting-key'

    key_file_path = os.path.join(parent_directory, parent_mount_path, secret_relative_path)

    if os.path.exists(key_file_path):
        with open(key_file_path, 'r') as key_file:
            key_data = key_file.read()
        key_json = json.loads(key_data)

        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = key_file_path
    else:
        print("Error: forecasting key file not found at", key_file_path)



    storage_client = storage.Client()
    bucket_name = "soccer-forecasting"
    bucket = storage_client.get_bucket(bucket_name)

    colors = ['#036666', '#14746F', '#248277', '#358F80','#ffffff' ,'#E5383B', '#BA181B', '#A4161A', '#660708']

    xpts_cm = LinearSegmentedColormap.from_list('xpts_preformance', colors, N=250)
    cm.register_cmap(name='xpts_preformance', cmap=xpts_cm)

    colors_odds = ['#d8f3dc', '#b7e4c7', '#95d5b2', '#74c69d', '#52b788', '#40916c', '#2d6a4f', '#1b4332', '#081c15'
                   ]
    odd_cm = LinearSegmentedColormap.from_list('ODD', colors_odds, N=250)
    cm.register_cmap(name='ODD', cmap=odd_cm)


    league_name='English Premier League'

    league_id = 47
    fixtures_data = get_league_fixtures(league_id)

    match_rounds = fixtures_data['matches']['allMatches']
    season_length = len(fixtures_data['matches']['allMatches'])

    today = datetime.date.today().strftime("%Y%m%d")
#%%
    def get_current_match_round(fixtures_data):
        if 'hasOngoingMatch' in fixtures_data and fixtures_data['hasOngoingMatch']:
            return "There is an ongoing match."

        if 'allMatches' in fixtures_data['matches']:
            last_finished_match_round = None
            for match in fixtures_data['matches']['allMatches']:
                if 'status' in match and match['status'].get('finished', False):
                    last_finished_match_round = match['round']

            if last_finished_match_round is not None:
                return last_finished_match_round

        return "No ongoing matches, but unable to determine the current match round."

    current_round = int(get_current_match_round(fixtures_data))


    if are_all_matches_finished(season_length, match_rounds, current_round):
        print(f"All matches in round {current_round - 1} are finished.")
    league_id = 47
    fixtures_data = get_league_fixtures(league_id)

    all_matches_data = fixtures_data['matches']['allMatches']

    df = pd.json_normalize(all_matches_data)
    df_v1=remove_aborted_matches(df)

    all_fixtures_df = df_v1[[
        'round', 'home.name', 'home.id', 'away.id', 'away.name', 'id'
    ]]
    all_fixtures_df = all_fixtures_df.rename(columns={
        'home.name': 'home_team_name', 'away.name': 'away_team_name', 'id': 'matchId','round':'matchRound',
        'home.id':'home_team_id','away.id':'away_team_id'
    })
    all_fixtures_df['matchRound'] = all_fixtures_df['matchRound'].astype(int)
    all_fixtures_df['matchId'] = all_fixtures_df['matchId'].astype(int)
    all_fixtures_df['home_team_id'] = all_fixtures_df['home_team_id'].astype(int)
    all_fixtures_df['away_team_id'] = all_fixtures_df['away_team_id'].astype(int)

    all_fixtures_df = all_fixtures_df.replace({'Tottenham': 'Tottenham Hotspur'})

    upcoming_games = all_fixtures_df[all_fixtures_df['matchRound'] == current_round]


    upcoming_games.loc[:, 'matchId'] = upcoming_games['matchId'].astype(int)

    if not is_model_built('Premier_League'):
        df_goals = get_season_matchgoals_data('2023/2024', 47)
        df_goals = df_goals.replace({'Tottenham': 'Tottenham Hotspur'})


        dc_model = Dixon_Coles_Model()
        model_params = dc_model.fit_poisson_model(df=df_goals, xi=0.0001)
        model_path = f'model/modelPremier_League.pkl'
        joblib.dump(model_params, model_path)
    else:
        model_path =  f'model/modelPremier_League.pkl'
        model_params = joblib.load(model_path)


    def simulate_match_current_round(match, upcoming_games, model_params, num_simulations):
        outcome_probs = iterate_k_simulations_upcoming_matches(match, upcoming_games, model_params, num_simulations)
        return outcome_probs

    # Number of parallel jobs, adjust as needed
    num_jobs = -1  # Use all available CPU cores, set to a specific number if needed

    # Perform simulations in parallel
    match_probs = Parallel(n_jobs=num_jobs)(
        delayed(simulate_match_current_round)(match, upcoming_games, model_params, 100)
        for match in upcoming_games['matchId']
    )

    # Print progress
    for index, match_prob in enumerate(match_probs):
        if index % 10 == 0:
            print(f'{index / len(match_probs):.1%} done.')

    # Convert the list of dictionaries to a DataFrame
    match_probs_df = pd.DataFrame(match_probs)

    # Merge the DataFrames
    merged_df = pd.merge(upcoming_games, match_probs_df, on='matchId')
    figure_buffer_matchround_forecast = BytesIO()
    matchround_forecast(df=merged_df, league='EPL', fotmob_leagueid=47,cmap='ODD').savefig(
        figure_buffer_matchround_forecast,
        format="png",
        dpi=600,
        bbox_inches="tight",
        edgecolor="none",
        transparent=False
    )

    figure_buffer_matchround_forecast.seek(0)
    blob_path_matchround_forecast = f"figures/{today}/matchround_forecast_{league_name}.png"
    blob_matchround_forecast = bucket.blob(blob_path_matchround_forecast)
    blob_matchround_forecast.upload_from_file(figure_buffer_matchround_forecast, content_type="image/png")
    figure_buffer_matchround_forecast.close()


    all_fixtures_df['matchRound'] = all_fixtures_df['matchRound'].astype(int)
    all_fixtures_df['matchId'] = all_fixtures_df['matchId'].astype(int)

    all_fixtures_df = all_fixtures_df.replace({'Tottenham': 'Tottenham Hotspur'})

    shots_data = get_latest_comp_shotsdata_parallel(season_length, league_id=47)
    remaining_matches = get_remaining_matches(shots_data, all_fixtures_df)
    remaining_matches['matchId'] = remaining_matches['matchId'].astype(int)
    shots_data['matchId'] = shots_data['matchId'].astype(int)



    shots_data['expectedGoals'] = shots_data['expectedGoals'].fillna(0)
    shots_data = shots_data.replace({'Tottenham': 'Tottenham Hotspur'})

    played_result,played_tables_drop_columns,unplayed_result,simulated_tables = run_simulations_parallel(20, shots_data, remaining_matches, model_params)

    played_result['matchId'] = played_result['matchId'].astype(int)

    played_tables_drop_columns['matchId'] = played_tables_drop_columns['matchId'].astype(int)
    unplayed_result['matchId'] = unplayed_result['matchId'].astype(int)

    results = pd.merge(played_tables_drop_columns, unplayed_result, how='outer',
                       left_on=['home_team_name', 'away_team_name', 'home_goals', 'away_goals', 'matchId'],
                       right_on=['home_team_name', 'away_team_name', 'home_goals', 'away_goals', 'matchId'
                                 ])

    sim_table = calculate_table(results)
    sim_table['team_id'] = sim_table['team'].map(get_team_id_mapping(league_id))
    updated_sim_table = calculate_per_90_metrics(sim_table)

    simulated_tables['team_id'] = simulated_tables['team'].map(get_team_id_mapping(league_id))

    position_prob = calculate_position_probabilities(simulated_tables)
    position_prob['team_id'] = position_prob['team'].map(get_team_id_mapping(league_id))

    xpts = calculate_xpts(played_result)
    xg_table = calculate_xg_table(shots_data)
    xpts = xg_table.merge(xpts, on='TeamName')

    table = get_league_table(league_id)
    xPoints_table = xpts.merge(table, on='TeamName')
    xPoints_table = xpoints_table_pre90_stats(xPoints_table)

    xPoints_table = xPoints_table.sort_values(by='pts', ascending=False)



    figure_buffer_xpt_table = BytesIO()
    xpt_table(xPoints_table, league_name='English Premier League').savefig(
        figure_buffer_xpt_table,
        format="png",
        dpi=600,
        bbox_inches="tight",
        edgecolor="none",
        transparent=False
    )
    figure_buffer_xpt_table.seek(0)



    figure_buffer_eos_distribution = BytesIO()
    eos_distribution_v1(simulated_tables, league_name).savefig(
        figure_buffer_eos_distribution,
        format="png",
        dpi=600,
        bbox_inches="tight",
        edgecolor="none",
        transparent=False
    )
    figure_buffer_eos_distribution.seek(0)

    figure_buffer_eos_table = BytesIO()
    plot_eos_table_v1(updated_sim_table, league_name).savefig(
        figure_buffer_eos_table,
        format="png",
        dpi=600,
        bbox_inches="tight",
        edgecolor="none",
        transparent=False
    )
    figure_buffer_eos_table.seek(0)

    figure_buffer_finishing_position = BytesIO()
    plot_finishing_position_distribution(position_prob, league_name).savefig(
        figure_buffer_finishing_position,
        format="png",
        dpi=600,
        bbox_inches="tight",
        edgecolor="none",
        transparent=False
    )
    figure_buffer_finishing_position.seek(0)


    blob_path_eos_distribution = f"figures/{today}/eos_distribution_{league_name}.png"
    blob_path_eos_table = f"figures/{today}/eos_table_{league_name}.png"
    blob_path_finishing_position = f"figures/{today}/finishing_position_odds_{league_name}.png"
    blob_path_xpt_table = f"figures/{today}/xpt_table_{league_name}.png"


    blob_eos_distribution = bucket.blob(blob_path_eos_distribution)
    blob_eos_distribution.upload_from_file(figure_buffer_eos_distribution, content_type="image/png")

    blob_eos_table = bucket.blob(blob_path_eos_table)
    blob_eos_table.upload_from_file(figure_buffer_eos_table, content_type="image/png")

    blob_finishing_position = bucket.blob(blob_path_finishing_position)
    blob_finishing_position.upload_from_file(figure_buffer_finishing_position, content_type="image/png")

    blob_xpt_table = bucket.blob(blob_path_xpt_table)
    blob_xpt_table.upload_from_file(figure_buffer_xpt_table, content_type="image/png")

    figure_buffer_eos_distribution.close()
    figure_buffer_eos_table.close()
    figure_buffer_finishing_position.close()
    figure_buffer_xpt_table.close()

if __name__ == "__main__":
    main()

