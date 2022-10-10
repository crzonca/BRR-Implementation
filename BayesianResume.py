import itertools
import math
import warnings

import choix
import colorcet as cc
import networkx as nx
import numpy as np
import pandas as pd
import scipy.integrate as integrate
import scipy.optimize as optimize
import scipy.stats as stats
from bokeh.models import ColumnDataSource, FixedTicker, NumeralTickFormatter
from bokeh.plotting import figure, show
from numba import njit
from numba_stats import norm
from prettytable import PrettyTable

save_dir = 'C:\\Users\\colin\\OneDrive\\Desktop\\Football\\'


def get_ranking_table(adjacency_file='2021NFL.csv', estimated_parity=2):
    df = pd.read_csv(save_dir + adjacency_file, index_col=0)

    bradley_terry_coeffs, pagerank = bradley_terry(df)
    league_parity, full_teams = bayesian_resume(df, league_parity=estimated_parity)
    print('League Parity:', round(league_parity, 3))
    print()

    column_names = ['Rank', 'Name', 'Wins', 'Losses', 'BT Coeff.', 'BRR', 'BRR Dev.', 'Pagerank']
    ranking_df = pd.DataFrame(columns=column_names)

    table = PrettyTable(column_names)
    table.float_format = '0.3'

    sorted_by_brr = sorted(full_teams, key=lambda tup: tup[3], reverse=True)
    for rank, team in enumerate(sorted_by_brr):
        row = list()
        row.append(rank + 1)
        team_info = list()
        team_info.append(team[0])
        team_info.append(round(team[1]))
        team_info.append(round(team[2]))
        team_info.append(bradley_terry_coeffs.get(team[0]))
        team_info.append(team[3])
        team_info.append(team[4])
        team_info.append(pagerank.get(team[0]))
        row = row + team_info

        ranking_df.loc[len(ranking_df.index)] = row
        table.add_row(row)

    print('Rankings')
    print(table)
    print()
    # ranking_df.to_csv(save_dir + '2021NFLRankings.csv', index=False)


def bradley_terry(df, ep=False, lsr=False, mm=False):
    teams = list(df.index)
    df = df.fillna(0)

    teams_to_index = {team: i for i, team in enumerate(teams)}
    index_to_teams = {i: team for team, i in teams_to_index.items()}

    graph = nx.from_pandas_adjacency(df, create_using=nx.DiGraph)
    edges = [list(itertools.repeat((teams_to_index.get(team2),
                                    teams_to_index.get(team1)),
                                   int(weight_dict.get('weight'))))
             for team1, team2, weight_dict in graph.edges.data()]
    edges = list(itertools.chain.from_iterable(edges))

    pagerank = nx.pagerank_numpy(graph)
    pagerank = {team: coeff * len(teams) for team, coeff in pagerank.items()}

    if not ep and not lsr and not mm:
        coeffs = pd.Series(choix.opt_pairwise(n_items=len(teams), data=edges))
        coeffs = coeffs.sort_values(ascending=False)
        coeffs = {index_to_teams.get(index): coeff for index, coeff in coeffs.iteritems()}
        return coeffs, pagerank
    if mm:
        coeffs = pd.Series(choix.mm_pairwise(n_items=len(teams), data=edges, alpha=1e-6))
        coeffs = coeffs.sort_values(ascending=False)
        coeffs = {index_to_teams.get(index): coeff for index, coeff in coeffs.iteritems()}
        return coeffs, pagerank
    if ep and not lsr and not lsr:
        coeffs, var = choix.ep_pairwise(n_items=len(teams), data=edges, alpha=1)
        var = pd.Series(np.diagonal(var))
        coeffs = pd.Series(coeffs)
        coeffs = pd.DataFrame({'Coeff': coeffs, 'Var': var})
        coeffs = coeffs.sort_values(by='Coeff', ascending=False)
        # coeffs = {index_to_teams.get(index): (coeff_tup['Coeff'], coeff_tup['Var'])
        #           for index, coeff_tup in coeffs.iterrows()}
        coeffs = {index_to_teams.get(index): coeff_tup['Coeff'] for index, coeff_tup in coeffs.iterrows()}
    if lsr:
        coeffs = pd.Series(choix.lsr_pairwise(n_items=len(teams), data=edges))
        coeffs = coeffs.sort_values(ascending=False)
        coeffs = {index_to_teams.get(index): coeff for index, coeff in coeffs.iteritems()}

    # df = pd.DataFrame(nx.to_numpy_array(graph), columns=teams)
    # df.index = teams

    return coeffs, pagerank


def get_probability_of_game_bt(coeff1, coeff2):
    return math.exp(coeff1) / (math.exp(coeff1) + math.exp(coeff2))


def bayesian_resume(wins, league_parity=2.0, max_iter=500, verbose=False):
    losses = wins.T

    teams = set(wins.index)

    wins_dict = dict()
    losses_dict = dict()
    for team in teams:
        team_wins = wins.loc[wins[team] > 0][team]
        team_wins_dict = dict(team_wins)
        wins_dict[team] = team_wins_dict

        team_losses = losses.loc[losses[team] > 0][team]
        team_losses_dict = dict(team_losses)
        losses_dict[team] = team_losses_dict

    teams = [(team, 0.0, 0.0) for team in teams]
    team_names = [team[0] for team in teams]

    all_matchups = list()
    for team_name in team_names:
        wins = wins_dict.get(team_name)

        opponent_wins_names = [name for name, record in wins.items() for name in itertools.repeat(name, int(record))]
        matchups = [(team_name, opponent_name) for opponent_name in opponent_wins_names]
        all_matchups.extend(matchups)

    continue_iteration = True
    final_iteration = False

    num_iterations = 0
    while continue_iteration and num_iterations <= max_iter:
        num_iterations = num_iterations + 1
        for team_name in team_names:
            wins = wins_dict.get(team_name)
            losses = losses_dict.get(team_name)

            opponent_wins_names = [name for name, record in wins.items() for name in
                                   itertools.repeat(name, int(record))]
            opponent_losses_names = [name for name, record in losses.items() for name in
                                     itertools.repeat(name, int(record))]
            opponent_names = opponent_wins_names + opponent_losses_names

            opponent_ratings = [get_matching_team(teams, team)[1] for team in opponent_names]
            opponent_devs = [get_matching_team(teams, team)[2] for team in opponent_names]
            team_record = list(itertools.repeat(1, len(opponent_wins_names))) + \
                          list(itertools.repeat(0, len(opponent_losses_names)))

            team_dev = get_matching_team(teams, team_name)[2]

            if len(opponent_ratings) == 0:
                team_brr = 0.0
                team_s = 1.0
            else:
                team_brr = get_brr(opponent_ratings, team_dev, opponent_devs, team_record, league_parity)
                team_s = get_brr_dev(team_brr, opponent_ratings, team_dev, opponent_devs, team_record, league_parity)

            teams = [(team[0], team[1], team[2]) for team in teams if team[0] != team_name]
            teams.append((team_name, team_brr, team_s))

        winner_brrs = [get_matching_team(teams, winner_name)[1] for winner_name, loser_name in all_matchups]
        loser_brrs = [get_matching_team(teams, loser_name)[1] for winner_name, loser_name in all_matchups]
        winner_devs = [get_matching_team(teams, winner_name)[2] for winner_name, loser_name in all_matchups]
        loser_devs = [get_matching_team(teams, loser_name)[2] for winner_name, loser_name in all_matchups]
        parity = get_league_parity(winner_brrs, loser_brrs, winner_devs, loser_devs)
        if verbose:
            print(num_iterations, parity)
        if parity == 0:
            parity = 1e-16
        if parity >= 1000:
            parity = 1000

        if final_iteration:
            continue_iteration = False

        if parity == league_parity:
            final_iteration = True
        league_parity = parity

    full_teams = list()
    for team in teams:
        team_wins = sum(wins_dict.get(team[0]).values())
        team_losses = sum(losses_dict.get(team[0]).values())
        full_teams.append((team[0], team_wins, team_losses, team[1], team[2]))

    if num_iterations > max_iter:
        print('Failed to Converge within', max_iter, 'iterations')
    return league_parity, full_teams


def get_matching_team(teams, team):
    matching_team = [t for t in teams if t[0] == team][0]
    return matching_team


@njit
def get_sigma_t(sigmas):
    sigmas = np.array(sigmas)
    return math.sqrt(np.sum(sigmas ** 2))


@njit
def get_probability_of_game_brr(brr1, brr2, sigmas, won):
    sigma_t = get_sigma_t(sigmas)
    upper_bound = (brr1 - brr2) / sigma_t

    p_x = norm.cdf(np.array([upper_bound]), 0.0, 1.0)[0]

    return abs(won + p_x - 1)


@njit
def get_probability_of_season(brr1, opp_brrs, sigma, opp_sigmas, record, league_parity):
    p_xs = list()
    for opp_brr, opp_sigma, won_game in zip(opp_brrs, opp_sigmas, record):
        sigmas = [league_parity, league_parity, sigma, opp_sigma]
        prob = get_probability_of_game_brr(brr1, opp_brr, sigmas, won_game)
        p_xs.append(prob)

    return np.prod(np.array(p_xs))


@njit
def brr_numerator_integrand(x, opp_brrs, sigma, opp_sigmas, record, league_parity):
    return x * norm.pdf(np.array([x]), 0.0, 1.0)[0] * get_probability_of_season(x, opp_brrs, sigma, opp_sigmas, record,
                                                                                league_parity)


@njit
def brr_s_numerator_integrand(x, brr, opp_brrs, sigma, opp_sigmas, record, league_parity):
    return math.pow(brr - x, 2) * norm.pdf(np.array([x]), 0.0, 1.0)[0] * get_probability_of_season(x, opp_brrs, sigma,
                                                                                                   opp_sigmas,
                                                                                                   record,
                                                                                                   league_parity)


@njit
def denominator_integrand(x, opp_brrs, sigma, opp_sigmas, record, league_parity):
    return norm.pdf(np.array([x]), 0.0, 1.0)[0] * get_probability_of_season(x, opp_brrs, sigma, opp_sigmas, record,
                                                                            league_parity)


def get_brr(opp_brrs, sigma, opp_sigmas, record, league_parity):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        numerator = integrate.quad(brr_numerator_integrand,
                                   a=-np.inf,
                                   b=np.inf,
                                   args=(opp_brrs, sigma, opp_sigmas, record, league_parity))[0]
        denominator = integrate.quad(denominator_integrand,
                                     a=-np.inf,
                                     b=np.inf,
                                     args=(opp_brrs, sigma, opp_sigmas, record, league_parity))[0]
    return numerator / denominator


def get_brr_dev(b, opp_brrs, sigma, opp_sigmas, record, league_parity):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        numerator = integrate.quad(brr_s_numerator_integrand,
                                   a=-np.inf,
                                   b=np.inf,
                                   args=(b, opp_brrs, sigma, opp_sigmas, record, league_parity))[0]
        denominator = integrate.quad(denominator_integrand,
                                     a=-np.inf,
                                     b=np.inf,
                                     args=(opp_brrs, sigma, opp_sigmas, record, league_parity))[0]
    return math.sqrt(numerator / denominator)


@njit
def parity_integrand(y, winner_brr, loser_brr, winner_dev, loser_dev, league_parity):
    first = norm.pdf(np.array([y]), (loser_brr - winner_brr), get_sigma_t([loser_dev, winner_dev]))[0]

    upper_bound = y / (league_parity * math.sqrt(2))
    inner_portion = norm.cdf(np.array([upper_bound]), 0.0, 1.0)[0]

    return first * math.pow(inner_portion, 2)


def parity_function(possible_parity, winner_brrs, loser_brrs, winner_devs, loser_devs):
    if possible_parity == 0:
        possible_parity = 1e-16
    ps = list()
    for winner_brr, loser_brr, winner_dev, loser_dev in zip(winner_brrs, loser_brrs, winner_devs, loser_devs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = integrate.quad(parity_integrand,
                               a=-np.inf,
                               b=np.inf,
                               args=(winner_brr, loser_brr, winner_dev, loser_dev, possible_parity))[0]

        ps.append(p)
    return sum(ps)


def get_league_parity(winner_brrs, loser_brrs, winner_devs, loser_devs):
    parity = optimize.minimize_scalar(fun=parity_function,
                                      tol=1e-3,
                                      options={'maxiter': 30},
                                      args=(winner_brrs, loser_brrs, winner_devs, loser_devs))
    if parity.nit == 30:
        print('Max iterations reached')
    return round(parity.x, 5)


def plot_normals(names, means, deviations):
    if len(means) != len(deviations) != len(names):
        pass

    data = list(zip(names, means, deviations))
    data = sorted(data, key=lambda tup: tup[1], reverse=False)

    num_schools = len(names)
    scale = 50 / num_schools
    if scale > 1.8:
        scale = 1.8
    palette = [cc.rainbow[i * int(256 / num_schools)] for i in range(num_schools)]

    min_x = min(means) - 2.5 * max(deviations)
    max_x = max(means) + 2.5 * max(deviations)

    x = np.linspace(min_x, max_x, 500)
    source = ColumnDataSource(data={'x': x})
    p = figure(y_range=[datum[0] for datum in data],
               width=1200,
               x_range=(min_x, max_x),
               toolbar_location=None)

    for i, data in enumerate(data):
        school, mean, deviation = data
        dist = stats.norm(loc=mean, scale=deviation)
        pdf_vals = dist.pdf(x) * scale
        source.add(list(zip([school] * len(pdf_vals), pdf_vals)), school)
        p.patch('x', school, color=palette[i], alpha=0.6, line_color='black', source=source)

    p.outline_line_color = None
    p.background_fill_color = '#efefef'

    p.xaxis.ticker = FixedTicker(ticks=list(range(int(min_x) - 1, int(max_x) + 1, 1)))
    p.xaxis.formatter = NumeralTickFormatter()

    p.ygrid.grid_line_color = None
    p.xgrid.grid_line_color = '#dddddd'
    p.xgrid.ticker = p.xaxis.ticker

    p.axis.minor_tick_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.axis_line_color = None

    p.y_range.range_padding = scale / 5

    show(p)
