import numpy as np
from scipy.stats import poisson

import pandas as pd
from tqdm import tqdm

def rho_correction(goals_home, goals_away, home_exp, away_exp, rho):
    if goals_home == 0 and goals_away == 0:
        return 1 - (home_exp * away_exp * rho)
    elif goals_home == 0 and goals_away == 1:
        return 1 + (home_exp * rho)
    elif goals_home == 1 and goals_away == 0:
        return 1 + (away_exp * rho)
    elif goals_home == 1 and goals_away == 1:
        return 1 - rho
    else:
        return 1.0

def predict(params, home_team, away_team, max_goals=6):
    home_attack = params["attack_" + home_team]
    home_defence = params["defence_" + home_team]
    away_attack = params["attack_" + away_team]
    away_defence = params["defence_" + away_team]
    home_advantage = params["home_adv"]
    rho = params["rho"]

    team_avgs = [
        np.exp(home_attack + away_defence + home_advantage),
        np.exp(away_attack + home_defence)
    ]

    team_pred = [
        [poisson.pmf(i, team_avg) for i in range(max_goals + 1)] for team_avg in team_avgs
    ]

    output_matrix = np.outer(np.array(team_pred[0]), np.array(team_pred[1]))

    correction_matrix = np.array([
        [rho_correction(home_goals, away_goals, team_avgs[0], team_avgs[1], rho) for away_goals in range(2)]
        for home_goals in range(2)
    ])

    output_matrix[:2, :2] = output_matrix[:2, :2] * correction_matrix

    # Normalize to percentages
    total_probability = np.sum(output_matrix)
    output_matrix_percentage = (output_matrix / total_probability) * 100

    return output_matrix_percentage


def simulate_match(matchId, model_params,upcoming_match_data, k=int):
    """
    Performs k simulations on a match, and returns the average goals and probability of different scorelines.
    """

    # Extract match data
    upcoming_match_data = upcoming_match_data[upcoming_match_data['matchId'] == matchId]
    home_team_name = upcoming_match_data['home_team_name'].iloc[0]
    away_team_name = upcoming_match_data['away_team_name'].iloc[0]

    # Initialize counters for goals
    home_goals = 0
    away_goals = 0

    # Initialize probability variable
    total_probability = 0

    # Perform k simulations
    for _ in range(k):

        # Simulate a match
        h, a, prob = simulate_single_match(model_params, home_team_name, away_team_name)

        # Update goal counters
        home_goals += h
        away_goals += a

        # Update total probability
        total_probability += prob

    # Calculate average goals
    avg_home_goals = round(home_goals / k)
    avg_away_goals = round(away_goals / k)


    # Calculate average probability
    avg_probability = total_probability / k

    return {'home_goals': avg_home_goals,'home_team_name':home_team_name ,'away_goals': avg_away_goals,'away_team_name':away_team_name ,'probability': avg_probability, 'matchId': matchId}


# def simulate_single_match(model_params, home_team, away_team):
#     """
#     Simulates a single match and returns the home and away goals and probability.
#     """
#     goal = 6
#
#     home_attack = model_params["attack_" + home_team]
#     home_defence = model_params["defence_" + home_team]
#     away_attack = model_params["attack_" + away_team]
#     away_defence = model_params["defence_" + away_team]
#     home_advantage = model_params["home_adv"]
#     rho = model_params["rho"]
#
#     home_goal_expectation = np.exp(home_attack + away_defence + home_advantage)
#     away_goal_expectation = np.exp(away_attack + home_defence)
#
#     home_poisson = poisson(home_goal_expectation)
#     away_poisson = poisson(away_goal_expectation)
#
#     # Sample from the Poisson distributions
#     home_goal = int(home_poisson.rvs())  # Convert to integer
#     away_goal = int(away_poisson.rvs())  # Convert to integer
#
#     # Ensure goals are non-negative
#     home_goal = max(0, home_goal)
#     away_goal = max(0, away_goal)
#
#     home_probs = home_poisson.pmf(range(goal))
#     away_probs = away_poisson.pmf(range(goal))
#
#     m = np.outer(home_probs, away_probs)
#
#     m[0, 0] *= 1 - home_goal_expectation * away_goal_expectation * rho
#     m[0, 1] *= 1 + home_goal_expectation * rho
#     m[1, 0] *= 1 + away_goal_expectation * rho
#     m[1, 1] *= 1 - rho
#
#     prob = np.sum(np.tril(m, -1))  # Sum of probabilities of different scorelines
#
#     return home_goal, away_goal, prob
#



# def simulate_single_match(model_params, home_team, away_team):
#     """
#     Simulates a single match and returns the home and away goals and probability.
#     """
#     goal = 6
#
#     home_attack = model_params["attack_" + home_team]
#     home_defence = model_params["defence_" + home_team]
#     away_attack = model_params["attack_" + away_team]
#     away_defence = model_params["defence_" + away_team]
#     home_advantage = model_params["home_adv"]
#     rho = model_params["rho"]
#
#     home_goal_expectation = np.exp(home_attack + away_defence + home_advantage)
#     away_goal_expectation = np.exp(away_attack + home_defence)
#
#     home_poisson = poisson(home_goal_expectation)
#     away_poisson = poisson(away_goal_expectation)
#
#     # Sample from the Poisson distributions
#     home_goal = int(home_poisson.rvs())  # Convert to integer
#     away_goal = int(away_poisson.rvs())  # Convert to integer
#
#     # Ensure goals are non-negative
#     home_goal = max(0, home_goal)
#     away_goal = max(0, away_goal)
#
#     home_probs = home_poisson.pmf(range(goal))
#     away_probs = away_poisson.pmf(range(goal))
#
#     m = np.outer(home_probs, away_probs)
#
#     m[0, 0] *= 1 - home_goal_expectation * away_goal_expectation * rho
#     m[0, 1] *= 1 + home_goal_expectation * rho
#     m[1, 0] *= 1 + away_goal_expectation * rho
#     m[1, 1] *= 1 - rho
#
#     prob = np.sum(np.tril(m, -1))  # Sum of probabilities of different scorelines
#
#     return home_goal, away_goal, prob


def simulate_single_match(model_params, home_team, away_team):
    goal = 6

    home_attack = model_params["attack_" + home_team]
    home_defence = model_params["defence_" + home_team]
    away_attack = model_params["attack_" + away_team]
    away_defence = model_params["defence_" + away_team]
    home_advantage = model_params["home_adv"]
    rho = model_params["rho"]

    home_goal_expectation = np.exp(home_attack + away_defence + home_advantage)
    away_goal_expectation = np.exp(away_attack + home_defence)

    home_poisson = poisson(home_goal_expectation)
    away_poisson = poisson(away_goal_expectation)

    # Create arrays of probabilities using the pmf method
    home_probs = home_poisson.pmf(np.arange(goal))
    away_probs = away_poisson.pmf(np.arange(goal))

    m = np.outer(home_probs, away_probs)

    m[0, 0] *= 1 - home_goal_expectation * away_goal_expectation * rho
    m[0, 1] *= 1 + home_goal_expectation * rho
    m[1, 0] *= 1 + away_goal_expectation * rho
    m[1, 1] *= 1 - rho

    prob = np.sum(np.tril(m, -1))  # Sum of probabilities of different scorelines

    # Calculate expected goals without rounding
    home_expected_goals = np.sum(np.arange(goal) * home_probs)
    away_expected_goals = np.sum(np.arange(goal) * away_probs)

    return home_expected_goals, away_expected_goals, prob





# def simulate_single_match_parallel(model_params, home_team_name, away_team_name):
#     """
#     Simulates a single match in parallel and returns the home and away goals and probability.
#     """
#     return simulate_single_match(model_params, home_team_name, away_team_name)

# def simulate_match_parallel(matchIds, model_params, upcoming_match_data, k=int):
#     """
#     Performs k simulations on multiple matches in parallel, returning the average goals and probability of different scorelines,
#     along with the detailed results of each simulation.
#     """
#     detailed_results = []
#
#     def single_simulation(matchId):
#         match_data = upcoming_match_data[upcoming_match_data['matchId'] == matchId]
#         home_team_name = match_data['home_team_name'].iloc[0]
#         away_team_name = match_data['away_team_name'].iloc[0]
#
#         match_results = [(*simulate_single_match_parallel(model_params, home_team_name, away_team_name), matchId) for _ in range(k)]
#         detailed_results.extend(match_results)
#         return match_results
#
#     # Perform simulations for each match in parallel
#     results = Parallel(n_jobs=-1)(delayed(single_simulation)(matchId) for matchId in matchIds)
#
#     # Initialize counters for goals and probability
#     total_home_goals, total_away_goals, total_probability = 0, 0, 0
#
#     # Aggregate results
#     for match_result in results:
#         for h, a, prob, matchId in match_result:
#             total_home_goals += h
#             total_away_goals += a
#             total_probability += prob
#
#     # Calculate average goals
#     avg_home_goals = round(total_home_goals / (k * len(matchIds)))
#     avg_away_goals = round(total_away_goals / (k * len(matchIds)))
#
#     # Calculate average probability
#     avg_probability = total_probability / (k * len(matchIds))
#
#     # Create a DataFrame for aggregated results
#     aggregated_results = pd.DataFrame({
#         'home_goals': [avg_home_goals],
#         'away_goals': [avg_away_goals],
#         'probability': [avg_probability],
#         'matchIds': [matchIds]
#     })
#
#     # Create a DataFrame for detailed results
#     detailed_results_df = pd.DataFrame(detailed_results, columns=['home_goals', 'away_goals', 'probability', 'matchId'])
#
#     return aggregated_results, detailed_results_df


from scipy.stats import poisson

# def simulate_goals(expected_goals, attack_param):
#     """
#     Simulate goals based on the expected goal values using Poisson distribution.
#     """
#     goals = 0
#     if expected_goals.shape[0] > 0:
#         for shot in expected_goals:
#             goals += poisson.rvs(mu=attack_param * shot)
#             goals = min(6, goals)  # Limit the maximum number of goals to 6
#     return max(0, goals)
#
# def simulate_match_on_shots_xg(matchId, shot_df, model_params, k):
#     """
#     This function takes a match ID and simulates an outcome based on the shots
#     taken by each team, using attack parameters from the model.
#
#     Args:
#     - matchId (int): The ID of the match.
#     - shot_df (DataFrame): DataFrame containing shot data for the match.
#     - params (dict): Dictionary containing model parameters.
#     - k (int): Number of simulations to perform.
#
#     Returns:
#     - list: List of dictionaries containing simulated results and probabilities for each simulation.
#     """
#
#     shots = shot_df[shot_df['matchId'] == matchId]
#
#     shots_home = shots[shots['Venue'] == 'Home']
#     shots_away = shots[shots['Venue'] == 'Away']
#     home_team_name = shots_home['TeamName'].values[0]
#     away_team_name = shots_away['TeamName'].values[0]
#
#     # Store results and probabilities for each simulation
#     simulation_results = []
#
#     # Perform k simulations
#     for _ in range(k):
#         # Simulate a match
#         home_goals = simulate_goals(shots_home['expectedGoals'].to_numpy(), model_params["attack_" + home_team_name])
#         away_goals = simulate_goals(shots_away['expectedGoals'].to_numpy(), model_params["attack_" + away_team_name])
#
#         # Update outcome counters
#         if home_goals == away_goals:
#             outcome = 'draw'
#         elif home_goals > away_goals:
#             outcome = 'home_win'
#         else:
#             outcome = 'away_win'
#
#         # Store simulation results and probabilities
#         simulation_results.append({'home_goals': home_goals, 'away_goals': away_goals, 'outcome': outcome})
#
#     # Calculate probabilities
#     home_prob = sum(1 for result in simulation_results if result['outcome'] == 'home_win') / k
#     draw_prob = sum(1 for result in simulation_results if result['outcome'] == 'draw') / k
#     away_prob = sum(1 for result in simulation_results if result['outcome'] == 'away_win') / k
#
#     # Simulate final goals for home and away teams (using the first simulation)
#     home_goals_final = simulate_goals(shots_home['expectedGoals'], model_params["attack_" + home_team_name])
#     away_goals_final = simulate_goals(shots_away['expectedGoals'], model_params["attack_" + away_team_name])
#
#     return {'home_goals': home_goals_final, 'home_team_name':home_team_name, 'away_goals': away_goals_final,'away_team_name':away_team_name , 'home_prob': home_prob,
#             'draw_prob': draw_prob, 'away_prob': away_prob, 'matchId': matchId}
#
#


from scipy.stats import poisson

def simulate_goals(expected_goals, attack_param):
    """
    Simulate goals based on the cumulative expected goal values using Poisson distribution.
    """
    goals = poisson.rvs(mu=attack_param * expected_goals.sum())
    goals = min(6, goals)  # Limit the maximum number of goals to 6
    return max(0, goals)

def simulate_match_on_cumulative_xg(matchId, shot_df, model_params, k):
    """
    This function takes a match ID and simulates an outcome based on the cumulative shots
    taken by each team, using attack parameters from the model.

    Args:
    - matchId (int): The ID of the match.
    - shot_df (DataFrame): DataFrame containing shot data for the match.
    - model_params (dict): Dictionary containing model parameters.
    - k (int): Number of simulations to perform.

    Returns:
    - dict: Dictionary containing simulated results and probabilities for the match.
    """

    shots = shot_df[shot_df['matchId'] == matchId]

    shots_home = shots[shots['Venue'] == 'Home']
    shots_away = shots[shots['Venue'] == 'Away']
    home_team_name = shots_home['TeamName'].values[0]
    away_team_name = shots_away['TeamName'].values[0]

    # Store results and probabilities for each simulation
    simulation_results = []

    # Perform k simulations
    for _ in range(k):
        # Simulate a match
        home_goals = simulate_goals(shots_home['expectedGoals'].to_numpy(), model_params["attack_" + home_team_name])
        away_goals = simulate_goals(shots_away['expectedGoals'].to_numpy(), model_params["attack_" + away_team_name])

        # Update outcome counters
        if home_goals == away_goals:
            outcome = 'draw'
        elif home_goals > away_goals:
            outcome = 'home_win'
        else:
            outcome = 'away_win'

        # Store simulation results and probabilities
        simulation_results.append({'home_goals': home_goals, 'away_goals': away_goals, 'outcome': outcome})

    # Calculate probabilities
    home_prob = sum(1 for result in simulation_results if result['outcome'] == 'home_win') / k
    draw_prob = sum(1 for result in simulation_results if result['outcome'] == 'draw') / k
    away_prob = sum(1 for result in simulation_results if result['outcome'] == 'away_win') / k

    # Simulate final goals for home and away teams (using the first simulation)
    home_goals_final = simulate_goals(shots_home['expectedGoals'].sum(), model_params["attack_" + home_team_name])
    away_goals_final = simulate_goals(shots_away['expectedGoals'].sum(), model_params["attack_" + away_team_name])

    return {'home_goals': home_goals_final, 'home_team_name': home_team_name, 'away_goals': away_goals_final, 'away_team_name': away_team_name, 'home_prob': home_prob,
            'draw_prob': draw_prob, 'away_prob': away_prob, 'matchId': matchId}


def simulate_match_on_shots_xg_parallel(matchId, shot_df, model_params, k):
    """
    Simulates outcomes based on shots and expected goals using joblib for parallel processing.

    Args:
    - matchId (int): The ID of the match.
    - shot_df (DataFrame): DataFrame containing shot data for the match.
    - model_params (dict): Dictionary containing model parameters.
    - k (int): Number of simulations to perform.

    Returns:
    - DataFrame: Aggregated results containing average goals and probabilities.
    - DataFrame: Detailed results containing goals, probabilities, and outcomes for each simulation.
    """
    shots = shot_df[shot_df['matchId'] == matchId]

    shots_home = shots[shots['Venue'] == 'Home']
    shots_away = shots[shots['Venue'] == 'Away']
    home_team_name = shots_home['TeamName'].values[0]
    away_team_name = shots_away['TeamName'].values[0]

    # Define a function to perform a single simulation
    def single_simulation(_):
        home_goals = simulate_goals(shots_home['expectedGoals'].to_numpy(), model_params["attack_" + home_team_name])
        away_goals = simulate_goals(shots_away['expectedGoals'].to_numpy(), model_params["attack_" + away_team_name])

        if home_goals == away_goals:
            outcome = 'draw'
        elif home_goals > away_goals:
            outcome = 'home_win'
        else:
            outcome = 'away_win'

        return {'home_goals': home_goals, 'away_goals': away_goals, 'outcome': outcome}

    # Perform k simulations in parallel
    simulation_results = Parallel(n_jobs=-1)(delayed(single_simulation)(_) for _ in range(k))

    # Calculate probabilities
    home_prob = sum(1 for result in simulation_results if result['outcome'] == 'home_win') / k
    draw_prob = sum(1 for result in simulation_results if result['outcome'] == 'draw') / k
    away_prob = sum(1 for result in simulation_results if result['outcome'] == 'away_win') / k

    # Simulate final goals for home and away teams (using the first simulation)
    home_goals_final = simulate_goals(shots_home['expectedGoals'], model_params["attack_" + home_team_name])
    away_goals_final = simulate_goals(shots_away['expectedGoals'], model_params["attack_" + away_team_name])

    # Create a DataFrame for detailed results
    detailed_results_df = pd.DataFrame(simulation_results)

    # Create a DataFrame for aggregated results
    aggregated_results = pd.DataFrame({
        'home_goals': [home_goals_final],
        'away_goals': [away_goals_final],
        'home_prob': [home_prob],
        'draw_prob': [draw_prob],
        'away_prob': [away_prob],
        'matchId': [matchId]
    })

    return aggregated_results, detailed_results_df


def iterate_k_simulations_on_match_id(matchId, shot_df,model_params, k=10):
    '''
    Performs k simulations on a match, and returns the probabilities of a win, loss, draw.
    '''
    shots = shot_df[shot_df['matchId'] == matchId]


    home_team_name = shots['TeamName'].values[0]
    away_team_name = shots['TeamName'].values[0]

    # Count the number of occurrences
    home_win = 0
    draw = 0
    away_win = 0

    # Perform k simulations
    for _ in range(k):
        # Simulate a match
        home_goals = simulate_goals(shots['expectedGoals'].to_numpy(), model_params["attack_" + home_team_name])
        away_goals = simulate_goals(shots['expectedGoals'].to_numpy(), model_params["attack_" + away_team_name])

        # Update outcome counters
        if home_goals == away_goals:
            draw += 1
        elif home_goals > away_goals:
            home_win += 1
        else:
            away_win += 1

    home_prob = home_win / k
    draw_prob = draw / k
    away_prob = away_win / k

    return {'home_prob': home_prob, 'away_prob': away_prob, 'draw_prob': draw_prob, 'matchId': matchId}



def iterate_k_simulations_on_match_id_v1(matchId, shot_df, model_params, k=1):
    '''
    Performs k simulations on a match, and returns the probabilities of a win, loss, draw, along with average goals and team names.
    '''
    shots = shot_df[shot_df['matchId'] == matchId]

    shots_home = shots[shots['Venue'] == 'Home']
    shots_away = shots[shots['Venue'] == 'Away']
    home_team_name = shots_home['TeamName'].values[0]
    away_team_name = shots_away['TeamName'].values[0]

    # Count the number of occurrences
    home_win = 0
    draw = 0
    away_win = 0

    # Initialize counters for goals
    home_goals_total = 0
    away_goals_total = 0

    # Perform k simulations
    for _ in range(k):
        # Simulate a match
        home_goals = simulate_goals(shots_home['expectedGoals'].to_numpy(), model_params["attack_" + home_team_name])
        away_goals = simulate_goals(shots_away['expectedGoals'].to_numpy(), model_params["attack_" + away_team_name])

        # Update outcome counters
        if home_goals == away_goals:
            draw += 1
        elif home_goals > away_goals:
            home_win += 1
        else:
            away_win += 1

        # Update goal counters
        home_goals_total += home_goals
        away_goals_total += away_goals

    # Calculate probabilities after the loop
    home_prob_final = home_win / k
    away_prob_final = away_win / k
    draw_prob_final = draw / k

    # Calculate average goals
    avg_home_goals = round(home_goals_total / k)
    avg_away_goals = round(away_goals_total / k)

    return {
        'home_prob': home_prob_final,
        'away_prob': away_prob_final,
        'draw_prob': draw_prob_final,
        'matchId': matchId,
        'home_goals': avg_home_goals,
        'away_goals': avg_away_goals,
        'home_team_name': home_team_name,
        'away_team_name': away_team_name
    }

def iterate_k_simulations_upcoming_matches(matchId, df,model_params, k=10):
    '''
    Performs k simulations on a match, and returns the probabilities of a win, loss, draw.
    '''
    match_data = df[df['matchId'] == matchId].iloc[0]
    home_team_name = match_data['home_team_name']
    away_team_name = match_data['away_team_name']

    # Count the number of occurrences
    home_win = 0
    draw = 0
    away_win = 0

    # Perform k simulations
    for _ in range(k):
        # Simulate a match
        # home_goals, away_goals = simulate_single_match(model_params, home_team_name, away_team_name)
        home_goals, away_goals, _ = simulate_single_match(model_params, home_team_name, away_team_name)

        # Update outcome counters
        if home_goals == away_goals:
            draw += 1
        elif home_goals > away_goals:
            home_win += 1
        else:
            away_win += 1

    home_prob = home_win / k
    draw_prob = draw / k
    away_prob = away_win / k

    return {'home_prob': home_prob, 'away_prob': away_prob, 'draw_prob': draw_prob, 'matchId': matchId}


# def run_simulations(n_simulations, shot_df, remaining_games,  model_params):
#     simulated_tables = []
#
#     for simulation_id in tqdm(range(1, n_simulations + 1)):
#         played_matches = []  # Initialize a list to store DataFrames for each played match in the simulation
#         unplayed_matches = []  # Initialize a list to store DataFrames for each unplayed match in the simulation
#
#         # Simulate played matches
#         for index, matchId in enumerate(shot_df['matchId']):
#             simulated_data = iterate_k_simulations_on_match_id_v1(matchId,shot_df, model_params, 1)
#             simulated_data['simulation_id'] = simulation_id
#             nested_data = pd.DataFrame([simulated_data])
#             nested_data = nested_data.drop(columns={'home_prob', 'draw_prob', 'away_prob'})
#             played_matches.append(nested_data)
#
#         # Simulate unplayed matches
#         for index, matchId in enumerate(remaining_games['matchId']):
#             simulated_data = simulate_match(matchId, model_params, remaining_games, 1)
#             simulated_data['simulation_id'] = simulation_id
#             nested_data = pd.DataFrame([simulated_data])
#             nested_data = nested_data.drop(columns={'probability'})
#             unplayed_matches.append(nested_data)
#
#         # Concatenate the DataFrames for all played matches and unplayed matches in the current simulation
#         simulated_tables.extend(played_matches)
#         simulated_tables.extend(unplayed_matches)
#
#     # Concatenate all DataFrames for all simulations
#     simulated_tables_df = pd.concat(simulated_tables, ignore_index=True)
#
#     # Calculate the league table based on simulation_id
#     simulated_tables = simulated_tables_df.groupby('simulation_id').apply(lambda x: calculate_table(x)).reset_index()
#
#     return simulated_tables,simulated_tables_df,played_matches



def calculate_position_probabilities(league_table):
    total_simulations = league_table['simulation_id'].nunique()

    # Assuming sim_table has columns 'team' and 'points', adjust accordingly
    team_avg_points = league_table.groupby('team')['points'].mean().reset_index()

    position_probabilities = league_table.groupby('team')['position'].value_counts(normalize=True).unstack(fill_value=0)
    position_probabilities *= 100
    position_probabilities = position_probabilities.reset_index()

    # Merge with team_avg_points to get average points
    position_probabilities = pd.merge(position_probabilities, team_avg_points, on='team')

    # Sort by average points in descending order
    position_probabilities = position_probabilities.sort_values(by='points', ascending=False)

    return position_probabilities



def calculate_xpts(df):
    expected_points = {}

    for match_id, match_df in df.groupby('matchId'):
        home_team_name = match_df['home_team_name'].iloc[0]
        away_team_name = match_df['away_team_name'].iloc[0]

        # Assuming you have match probabilities somewhere in your dataframe, adjust as needed
        home_prob = match_df['home_prob'].iloc[0]
        away_prob = match_df['away_prob'].iloc[0]
        draw_prob = match_df['draw_prob'].iloc[0]

        home_points = home_prob * 3 + draw_prob * 1
        away_points = away_prob * 3 + draw_prob * 1

        # Update the expected points for home team
        if home_team_name in expected_points:
            expected_points[home_team_name] += home_points
        else:
            expected_points[home_team_name] = home_points

        # Update the expected points for away team
        if away_team_name in expected_points:
            expected_points[away_team_name] += away_points
        else:
            expected_points[away_team_name] = away_points

    # Create a DataFrame for both home and away teams
    combined_df = pd.DataFrame({'TeamName': list(expected_points.keys()), 'points': list(expected_points.values())})

    return combined_df

def calculate_table(df):
    home_teams = []
    away_teams = []
    home_goals_for = []
    home_goals_against = []
    home_points = []
    away_goals_for = []
    away_goals_against = []
    away_points = []

    for matchId in df['matchId'].unique():
        match_df = df[df['matchId'] == matchId]

        home_team_name = match_df['home_team_name'].iloc[0]
        away_team_name = match_df['away_team_name'].iloc[0]

        home_goals = match_df['home_goals'].iloc[0]
        away_goals = match_df['away_goals'].iloc[0]

        home_team_points = 3 if home_goals > away_goals else 1 if home_goals == away_goals else 0
        away_team_points = 3 if away_goals > home_goals else 1 if home_goals == away_goals else 0

        home_teams.append(home_team_name)
        away_teams.append(away_team_name)
        home_goals_for.append(home_goals)
        home_goals_against.append(away_goals)
        home_points.append(home_team_points)
        away_goals_for.append(away_goals)
        away_goals_against.append(home_goals)
        away_points.append(away_team_points)

    # Create DataFrames for home and away teams
    home_df = pd.DataFrame({'team': home_teams, 'gf': home_goals_for, 'ga': home_goals_against, 'points': home_points})
    away_df = pd.DataFrame({'team': away_teams, 'gf': away_goals_for, 'ga': away_goals_against, 'points': away_points})

    # Calculate other statistics for home and away teams
    home_stats = home_df.groupby('team').agg(
        w=('points', lambda x: (x == 3).sum()),
        d=('points', lambda x: (x == 1).sum()),
        l=('points', lambda x: (x == 0).sum()),
        gf=('gf', 'sum'),
        ga=('ga', 'sum')
    ).reset_index()

    away_stats = away_df.groupby('team').agg(
        w=('points', lambda x: (x == 3).sum()),
        d=('points', lambda x: (x == 1).sum()),
        l=('points', lambda x: (x == 0).sum()),
        gf=('gf', 'sum'),
        ga=('ga', 'sum')
    ).reset_index()

    # Combine home and away statistics
    stats_df = pd.DataFrame({
        'team': home_stats['team'],
        'w': home_stats['w'] + away_stats['w'],
        'd': home_stats['d'] + away_stats['d'],
        'l': home_stats['l'] + away_stats['l'],
        'gf': home_stats['gf'] + away_stats['gf'],
        'ga': home_stats['ga'] + away_stats['ga']
    })

    # Calculate goal difference and points
    stats_df['gd'] = stats_df['gf'] - stats_df['ga']
    stats_df['points'] = 3 * stats_df['w'] + stats_df['d']

    # Sort the DataFrame
    stats_df = stats_df.sort_values(by=['points', 'gd', 'gf'], ascending=[False, False, False])

    # Add position column
    stats_df['position'] = range(1, len(stats_df) + 1)

    return stats_df

from joblib import Parallel, delayed
from tqdm import tqdm

# def run_simulations_parallel(n_simulations, shot_df, remaining_games, model_params):
#
#     def simulate_played_match(matchId, shot_df, model_params, simulation_id):
#         simulated_data = iterate_k_simulations_on_match_id_v1(matchId, shot_df, model_params, 1)
#         simulated_data['simulation_id'] = simulation_id
#         nested_data = pd.DataFrame([simulated_data])
#         nested_data_drop_columns = nested_data.drop(columns={'home_prob', 'draw_prob', 'away_prob'})
#         return nested_data, nested_data_drop_columns
#
#     def simulate_unplayed_match(matchId, model_params, remaining_games, simulation_id):
#         simulated_data = simulate_match(matchId, model_params, remaining_games, 1)
#         simulated_data['simulation_id'] = simulation_id
#         nested_data = pd.DataFrame([simulated_data])
#         nested_data = nested_data.drop(columns={'probability'})
#         return nested_data
#
#     def run_single_simulation(simulation_id):
#         played_matches = []
#         played_matches_drop_columns = []
#         unplayed_matches = []
#
#         for matchId in tqdm(shot_df['matchId']):
#             played_match, played_match_drop_columns = simulate_played_match(matchId, shot_df, model_params, simulation_id)
#             played_matches.append(played_match)
#             played_matches_drop_columns.append(played_match_drop_columns)
#
#         for matchId in tqdm(remaining_games['matchId']):
#             unplayed_matches.append(simulate_unplayed_match(matchId, model_params, remaining_games, simulation_id))
#
#         played_tables = pd.concat(played_matches, ignore_index=True)
#         played_tables_drop_columns = pd.concat(played_matches_drop_columns, ignore_index=True)
#         unplayed_tables = pd.concat(unplayed_matches, ignore_index=True)
#
#         return played_tables, played_tables_drop_columns, unplayed_tables
#
#     played_tables_list, played_tables_drop_columns_list, unplayed_tables_list = zip(*Parallel(n_jobs=-1)(delayed(run_single_simulation)(simulation_id)
#                                                                                                          for simulation_id in range(1, n_simulations + 1)))
#
#     played_tables = pd.concat(played_tables_list, ignore_index=True)
#     played_tables_drop_columns = pd.concat(played_tables_drop_columns_list, ignore_index=True)
#     unplayed_tables = pd.concat(unplayed_tables_list, ignore_index=True)
#
#     # Concatenate played_tables_drop_columns with unplayed_tables
#     simulated_tables = pd.concat([played_tables_drop_columns, unplayed_tables], ignore_index=True)
#
#     # Calculate the league table based on simulation_id
#     simulated_tables = simulated_tables.groupby('simulation_id').apply(lambda x: calculate_table(x)).reset_index()
#
#     return played_tables, played_tables_drop_columns, unplayed_tables, simulated_tables


# def run_simulations_parallel(n_simulations, shot_df, remaining_games, model_params):
#     def simulate_played_match(matchId, shot_df, model_params, simulation_id):
#         simulated_data = iterate_k_simulations_on_match_id_v1(matchId, shot_df, model_params, 1)
#         simulated_data['simulation_id'] = simulation_id
#         nested_data = pd.DataFrame([simulated_data])
#         nested_data_drop_columns = nested_data.drop(columns={'home_prob', 'draw_prob', 'away_prob'})
#         return nested_data, nested_data_drop_columns
#
#     def simulate_unplayed_match(matchId, model_params, remaining_games, simulation_id):
#         simulated_data = simulate_match(matchId, model_params, remaining_games, 1)
#         simulated_data['simulation_id'] = simulation_id
#         nested_data = pd.DataFrame([simulated_data])
#         nested_data = nested_data.drop(columns={'probability'})
#         return nested_data
#
#     def run_single_simulation(simulation_id):
#         played_matches = []
#         played_matches_drop_columns = []
#         unplayed_matches = []
#
#         for index, matchId in enumerate(tqdm(shot_df['matchId'])):
#             played_match, played_match_drop_columns = simulate_played_match(matchId, shot_df, model_params, simulation_id)
#             played_matches.append(played_match)
#             played_matches_drop_columns.append(played_match_drop_columns)
#
#             # Print progress every 10 iterations
#             if index % 10 == 0:
#                 print(f'{index / len(shot_df) * 100:.1f}% done for played matches.')
#
#         for index, matchId in enumerate(tqdm(remaining_games['matchId'])):
#             unplayed_matches.append(simulate_unplayed_match(matchId, model_params, remaining_games, simulation_id))
#
#             # Print progress every 10 iterations
#             if index % 10 == 0:
#                 print(f'{index / len(remaining_games) * 100:.1f}% done for unplayed matches.')
#
#         played_tables = pd.concat(played_matches, ignore_index=True)
#         played_tables_drop_columns = pd.concat(played_matches_drop_columns, ignore_index=True)
#         unplayed_tables = pd.concat(unplayed_matches, ignore_index=True)
#
#         return played_tables, played_tables_drop_columns, unplayed_tables
#
#     played_tables_list, played_tables_drop_columns_list, unplayed_tables_list = zip(
#         *Parallel(n_jobs=-1)(delayed(run_single_simulation)(simulation_id) for simulation_id in range(1, n_simulations + 1)))
#
#     played_tables = pd.concat(played_tables_list, ignore_index=True)
#     played_tables_drop_columns = pd.concat(played_tables_drop_columns_list, ignore_index=True)
#     unplayed_tables = pd.concat(unplayed_tables_list, ignore_index=True)
#
#     # Concatenate played_tables_drop_columns with unplayed_tables
#     simulated_tables = pd.concat([played_tables_drop_columns, unplayed_tables], ignore_index=True)
#
#     # Calculate the league table based on simulation_id
#     simulated_tables = simulated_tables.groupby('simulation_id').apply(lambda x: calculate_table(x)).reset_index()
#
#     return played_tables, played_tables_drop_columns, unplayed_tables, simulated_tables


import pandas as pd
from joblib import Parallel, delayed

import pandas as pd
from joblib import Parallel, delayed

# def run_simulations_parallel(n_simulations, shot_df, remaining_games, model_params, batch_size=10):
#     def simulate_played_matches(match_ids, shot_df, model_params, simulation_id):
#         simulated_data_list = []
#         for matchId in match_ids:
#             simulated_data = iterate_k_simulations_on_match_id_v1(matchId, shot_df, model_params, 1)
#             simulated_data['simulation_id'] = simulation_id
#             simulated_data_list.append(simulated_data)
#         nested_data = pd.DataFrame(simulated_data_list)
#         nested_data_drop_columns = nested_data.drop(columns={'home_prob', 'draw_prob', 'away_prob'})
#         return nested_data, nested_data_drop_columns
#
#     def simulate_unplayed_matches(match_ids, model_params, remaining_games, simulation_id):
#         simulated_data_list = []
#         for matchId in match_ids:
#             simulated_data = simulate_match(matchId, model_params, remaining_games, 1)
#             simulated_data['simulation_id'] = simulation_id
#             simulated_data_list.append(simulated_data)
#         nested_data = pd.DataFrame(simulated_data_list)
#         nested_data = nested_data.drop(columns={'probability'})
#         return nested_data
#
#     def run_single_simulation(simulation_id, shot_df, remaining_games, model_params):
#         played_matches = []
#         played_matches_drop_columns = []
#         unplayed_matches = []
#
#         # Simulate played matches in batches
#         for i in range(0, len(shot_df), batch_size):
#             match_ids_batch = shot_df['matchId'].iloc[i:i+batch_size].tolist()
#             played_match, played_match_drop_columns = simulate_played_matches(match_ids_batch, shot_df, model_params, simulation_id)
#             played_matches.append(played_match)
#             played_matches_drop_columns.append(played_match_drop_columns)
#
#             # Print progress for played matches
#             print(f'Simulation {simulation_id}: {i / len(shot_df) * 100:.1f}% done for played matches.')
#
#         # Simulate unplayed matches in batches
#         for i in range(0, len(remaining_games), batch_size):
#             match_ids_batch = remaining_games['matchId'].iloc[i:i+batch_size].tolist()
#             unplayed_matches.append(simulate_unplayed_matches(match_ids_batch, model_params, remaining_games, simulation_id))
#
#             # Print progress for unplayed matches
#             print(f'Simulation {simulation_id}: {i / len(remaining_games) * 100:.1f}% done for unplayed matches.')
#
#         played_tables = pd.concat(played_matches, ignore_index=True)
#         played_tables_drop_columns = pd.concat(played_matches_drop_columns, ignore_index=True)
#         unplayed_tables = pd.concat(unplayed_matches, ignore_index=True)
#
#         return played_tables, played_tables_drop_columns, unplayed_tables
#
#
#     def process_simulation(simulation_id):
#         played_tables, played_tables_drop_columns, unplayed_tables = run_single_simulation(simulation_id, shot_df, remaining_games, model_params)
#
#         # Concatenate played_tables_drop_columns with unplayed_tables
#         simulated_tables = pd.concat([played_tables_drop_columns, unplayed_tables], ignore_index=True)
#
#         # Calculate the league table based on simulation_id
#         simulated_tables = simulated_tables.groupby('simulation_id').apply(lambda x: calculate_table(x)).reset_index()
#
#         return played_tables, played_tables_drop_columns, unplayed_tables, simulated_tables
#
#     # Run simulations in parallel
#     results = Parallel(n_jobs=-1)(delayed(process_simulation)(simulation_id) for simulation_id in range(1, n_simulations + 1))
#
#     played_tables_list, played_tables_drop_columns_list, unplayed_tables_list, simulated_tables_list = zip(*results)
#
#     played_tables = pd.concat(played_tables_list, ignore_index=True)
#     played_tables_drop_columns = pd.concat(played_tables_drop_columns_list, ignore_index=True)
#     unplayed_tables = pd.concat(unplayed_tables_list, ignore_index=True)
#     simulated_tables = pd.concat(simulated_tables_list, ignore_index=True)
#
#     return played_tables, played_tables_drop_columns, unplayed_tables, simulated_tables


# def run_simulations_parallel(n_simulations, shot_df, remaining_games, model_params, batch_size=10):
#     def simulate_played_matches(match_ids, shot_df, model_params, simulation_id):
#         simulated_data_list = []
#         for matchId in match_ids:
#             simulated_data = iterate_k_simulations_on_match_id_v1(matchId, shot_df, model_params, 1)
#             simulated_data['simulation_id'] = simulation_id
#             simulated_data_list.append(simulated_data)
#         nested_data = pd.DataFrame(simulated_data_list)
#         nested_data_drop_columns = nested_data.drop(columns={'home_prob', 'draw_prob', 'away_prob'})
#         return nested_data, nested_data_drop_columns
#
#     def simulate_unplayed_matches(match_ids, model_params, remaining_games, simulation_id):
#         simulated_data_list = []
#         for matchId in match_ids:
#             simulated_data = simulate_match(matchId, model_params, remaining_games, 1)
#             simulated_data['simulation_id'] = simulation_id
#             simulated_data_list.append(simulated_data)
#         nested_data = pd.DataFrame(simulated_data_list)
#         nested_data = nested_data.drop(columns={'probability'})
#         return nested_data
#
#     def run_single_simulation(simulation_id, shot_df, remaining_games, model_params):
#         played_matches = []
#         played_matches_drop_columns = []
#         unplayed_matches = []
#
#         # Simulate played and unplayed matches in parallel
#         played_match_ids_batches = [shot_df['matchId'].iloc[i:i+batch_size].tolist() for i in range(0, len(shot_df), batch_size)]
#         unplayed_match_ids_batches = [remaining_games['matchId'].iloc[i:i+batch_size].tolist() for i in range(0, len(remaining_games), batch_size)]
#
#         played_results = Parallel(n_jobs=-1)(delayed(simulate_played_matches)(match_ids, shot_df, model_params, simulation_id) for match_ids in played_match_ids_batches)
#         unplayed_results = Parallel(n_jobs=-1)(delayed(simulate_unplayed_matches)(match_ids, model_params, remaining_games, simulation_id) for match_ids in unplayed_match_ids_batches)
#
#         for played_result, unplayed_result in zip(played_results, unplayed_results):
#             played_matches.append(played_result[0])
#             played_matches_drop_columns.append(played_result[1])
#             unplayed_matches.append(unplayed_result)
#
#         # Print progress for played and unplayed matches
#         print(f'Simulation {simulation_id}: 100.0% done for played and unplayed matches.')
#
#         played_tables = pd.concat(played_matches, ignore_index=True)
#         played_tables_drop_columns = pd.concat(played_matches_drop_columns, ignore_index=True)
#         unplayed_tables = pd.concat(unplayed_matches, ignore_index=True)
#
#         return played_tables, played_tables_drop_columns, unplayed_tables
#
#     def process_simulation(simulation_id):
#         played_tables, played_tables_drop_columns, unplayed_tables = run_single_simulation(simulation_id, shot_df, remaining_games, model_params)
#
#         # Concatenate played_tables_drop_columns with unplayed_tables
#         simulated_tables = pd.concat([played_tables_drop_columns, unplayed_tables], ignore_index=True)
#
#         # Calculate the league table based on simulation_id
#         simulated_tables = simulated_tables.groupby('simulation_id').apply(lambda x: calculate_table(x)).reset_index()
#
#         return played_tables, played_tables_drop_columns, unplayed_tables, simulated_tables
#
#     # Run simulations in parallel
#     results = Parallel(n_jobs=-1)(delayed(process_simulation)(simulation_id) for simulation_id in range(1, n_simulations + 1))
#
#     played_tables_list, played_tables_drop_columns_list, unplayed_tables_list, simulated_tables_list = zip(*results)
#
#     played_tables = pd.concat(played_tables_list, ignore_index=True)
#     played_tables_drop_columns = pd.concat(played_tables_drop_columns_list, ignore_index=True)
#     unplayed_tables = pd.concat(unplayed_tables_list, ignore_index=True)
#     simulated_tables = pd.concat(simulated_tables_list, ignore_index=True)
#
#     return played_tables, played_tables_drop_columns, unplayed_tables, simulated_tables


# class SimulationRunner:
#     def __init__(self, n_simulations, n_jobs=1, batch_size=10, simulate_match_on_shots_xg=None, simulate_match=None):
#         self.n_simulations = n_simulations
#         self.n_jobs = n_jobs
#         self.batch_size = batch_size
#         self.simulate_match_on_shots_xg = simulate_match_on_shots_xg
#         self.simulate_match = simulate_match
#
#
#     def run_simulation_batch(self, batch_id, shot_df, remaining_games, model_params):
#         simulated_tables = []
#         played_matches_df_list = []
#
#         start_index = batch_id * self.batch_size
#         end_index = min((batch_id + 1) * self.batch_size, self.n_simulations)
#
#         for simulation_id in range(start_index, end_index):
#             played_matches = []
#             unplayed_matches = []
#
#             # Simulate played matches
#             for index, matchId in enumerate(shot_df['matchId']):
#                 simulated_data_played = list(self.simulate_match_on_shots_xg(matchId, shot_df, model_params, 1))
#
#                 simulated_data_played['simulation_id'] = simulation_id
#                 nested_data = pd.DataFrame([simulated_data_played])
#                 played_matches.append(nested_data)
#
#             # Simulate unplayed matches
#             for index, matchId in enumerate(remaining_games['matchId']):
#                 simulated_data = self.simulate_match(matchId, model_params, remaining_games, 1)
#                 simulated_data['simulation_id'] = simulation_id
#                 nested_data = pd.DataFrame([simulated_data])
#                 unplayed_matches.append(nested_data)
#
#             played_matches_df_list.extend(played_matches)
#             simulated_tables.extend([match.drop(columns=['home_prob', 'draw_prob', 'away_prob']) for match in played_matches])
#             simulated_tables.extend(unplayed_matches)
#
#         return simulated_tables, played_matches_df_list
#
#     def run_simulations(self, shot_df, remaining_games, model_params):
#         simulated_tables = []
#         played_matches_df_list = []
#
#         # Run simulations in parallel batches
#         results = Parallel(n_jobs=self.n_jobs)(
#             delayed(self.run_simulation_batch)(batch_id, shot_df.copy(), remaining_games.copy(), model_params)
#             for batch_id in range((self.n_simulations + self.batch_size - 1) // self.batch_size)
#         )
#
#         for result in results:
#             batch_simulated_tables, batch_played_matches_df_list = result
#             simulated_tables.extend(batch_simulated_tables)
#             played_matches_df_list.extend(batch_played_matches_df_list)
#
#         # Concatenate all DataFrames for all simulations
#         simulated_tables_df = pd.concat(simulated_tables, ignore_index=True)
#
#         # Calculate the league table based on simulation_id
#         simulated_tables = simulated_tables_df.groupby('simulation_id').apply(lambda x: calculate_table(x)).reset_index()
#
#         # Concatenate all DataFrames for played matches in all simulations
#         played_matches_df = pd.concat(played_matches_df_list, ignore_index=True)
#
#         return simulated_tables, simulated_tables_df, played_matches_df
#
#
#

#%%
from scipy.stats import poisson
import numpy as np

def simulate_match_on_shots_with_model(matchId, shot_df, model_params, max_goals=10):
    '''
    This function takes a match ID and simulates an outcome based on the shots
    taken by each team and the provided model parameters.

    Parameters:
    - matchId: The match ID for which to simulate the outcome
    - shot_df: DataFrame containing shots data
    - model_params: Dictionary containing model parameters (attack and defense strengths)
    - max_goals: Maximum number of goals to simulate (optional, default is 10)

    Returns:
    - result: Dictionary containing simulated home and away goals
    '''

    shots = shot_df[shot_df['matchId'] == matchId]

    # Extract shots for home and away teams
    shots_home = shots[shots['Venue'] == 'Home']
    shots_away = shots[shots['Venue'] == 'Away']
    home_team_name = shots_home['TeamName'].values[0]
    away_team_name = shots_away['TeamName'].values[0]

    # Simulate home and away goals using Poisson distribution
    home_goal_expectation = np.exp(
        model_params["attack_" + home_team_name] + model_params["defence_" + away_team_name] + model_params["home_adv"]
    )
    home_poisson = poisson(home_goal_expectation)
    home_goals = min(home_poisson.rvs(size=1).sum(), max_goals)

    away_goal_expectation = np.exp(
        model_params["attack_" + away_team_name] + model_params["defence_" + home_team_name]
    )
    away_poisson = poisson(away_goal_expectation)
    away_goals = min(away_poisson.rvs(size=1).sum(), max_goals)

    return {'home_goals': home_goals, 'away_goals': away_goals}

#%%
def iterate_k_simulations(match_id, shot_df, model_params, k=10000):
    '''
    Performs k simulations on a match, and returns the probabilities of a win, loss, draw,
    along with the most likely home and away goals.
    '''
    # Count the number of occurrences
    home = 0
    draw = 0
    away = 0
    scorelines = {}

    shot_df = shot_df[shot_df['matchId'] == match_id]

    # Get the teams
    home_team_name, away_team_name = shot_df.loc[shot_df['Venue'] == 'Home', 'TeamName'].values[0], shot_df.loc[shot_df['Venue'] == 'Away', 'TeamName'].values[0]

    for i in range(k):
        simulation = simulate_match_on_shots_with_model(match_id, shot_df, model_params, 6)

        if simulation['home_goals'] > simulation['away_goals']:
            home += 1
        elif simulation['home_goals'] < simulation['away_goals']:
            away += 1
        else:
            draw += 1

        scoreline = f"{simulation['home_goals']} - {simulation['away_goals']}"
        scorelines[scoreline] = scorelines.get(scoreline, 0) + 1

    home_prob = home / k
    draw_prob = draw / k
    away_prob = away / k

    # Find the most likely scoreline
    most_likely_scoreline = max(scorelines, key=scorelines.get)

    # Extract home and away goals from the most likely scoreline
    most_likely_home_goals, most_likely_away_goals = map(int, most_likely_scoreline.split(' - '))

    return {'home_team_name': home_team_name, 'away_team_name': away_team_name,
            'home_prob': home_prob, 'away_prob': away_prob, 'draw_prob': draw_prob,
            'home_goals': most_likely_home_goals,
            'away_goals': most_likely_away_goals,
            'matchId': match_id}


def run_simulations_parallel(n_simulations, shot_df, remaining_games, model_params):

    def simulate_played_match(matchId, shot_df, model_params, simulation_id):
        simulated_data = iterate_k_simulations(matchId, shot_df, model_params, 1)
        simulated_data['simulation_id'] = simulation_id
        nested_data = pd.DataFrame([simulated_data])
    
        nested_data_drop_columns = nested_data.drop(columns=['home_prob', 'draw_prob', 'away_prob'])
        return nested_data, nested_data_drop_columns


    def simulate_unplayed_match(matchId, model_params, remaining_games, simulation_id):
        simulated_data = simulate_match(matchId, model_params, remaining_games, 1)
        simulated_data['simulation_id'] = simulation_id
        nested_data = pd.DataFrame([simulated_data])
        nested_data = nested_data.drop(columns={'probability'})
        return nested_data

    def run_single_simulation(simulation_id):
        played_matches = []
        played_matches_drop_columns = []
        unplayed_matches = []

        for matchId in tqdm(shot_df['matchId']):
            played_match, played_match_drop_columns = simulate_played_match(matchId, shot_df, model_params, simulation_id)
            played_matches.append(played_match)
            played_matches_drop_columns.append(played_match_drop_columns)

        for matchId in tqdm(remaining_games['matchId']):
            unplayed_matches.append(simulate_unplayed_match(matchId, model_params, remaining_games, simulation_id))

        played_tables = pd.concat(played_matches, ignore_index=True)
        played_tables_drop_columns = pd.concat(played_matches_drop_columns, ignore_index=True)
        unplayed_tables = pd.concat(unplayed_matches, ignore_index=True)

        return played_tables, played_tables_drop_columns, unplayed_tables

    played_tables_list, played_tables_drop_columns_list, unplayed_tables_list = zip(*Parallel(n_jobs=-1)(delayed(run_single_simulation)(simulation_id)
                                                                                                         for simulation_id in range(1, n_simulations + 1)))

    played_tables = pd.concat(played_tables_list, ignore_index=True)
    played_tables_drop_columns = pd.concat(played_tables_drop_columns_list, ignore_index=True)
    unplayed_tables = pd.concat(unplayed_tables_list, ignore_index=True)

    # Concatenate played_tables_drop_columns with unplayed_tables
    simulated_tables = pd.concat([played_tables_drop_columns, unplayed_tables], ignore_index=True)

    # Calculate the league table based on simulation_id
    simulated_tables = simulated_tables.groupby('simulation_id').apply(lambda x: calculate_table(x)).reset_index()

    return played_tables, played_tables_drop_columns, unplayed_tables, simulated_tables

