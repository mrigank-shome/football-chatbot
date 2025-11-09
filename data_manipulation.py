import pandas as pd
import string
from string import digits

df = pd.read_csv('data/events.csv')

print('The number of events is ' + str(len(df['id_odsp']))  + ' in ' + str(len(df['id_odsp'].unique())) + ' games.')

with open('data/dictionary.txt','r') as f:
    data_dict = f.read().strip()

data_dict = data_dict.split('\n\n\n')
data_dict = [i.split('\n') for i in data_dict]
data_dict = {data_dict[i][0]:data_dict[i][1:] for i in range(len(data_dict))}
data_dict = {key:[i.translate({ord(char): None for char in digits}).strip() for i in value] for key,value in data_dict.items()}

with open('data/teams.txt') as f:
    wanted_teams = f.read().splitlines()

df = df[df['event_team'].isin(wanted_teams)]
df = df[df['opponent'].isin(wanted_teams)]

print('The number of events is ' + str(len(df['id_odsp']))  + ' in ' 
      + str(len(df['id_odsp'].unique())) + ' games.')

players = []

for i in range(len(df) - 1):
    temp_player = df.iloc[i,10]
    if temp_player not in players:
        players.append(temp_player)

df_goal = df[df['is_goal'] != 0]
df_goal.dropna()

player_goals = pd.DataFrame(columns = ['player', 'goals'])

for player in players:
    df_temp = df_goal[df_goal['player'] == player]
    temp_player = player
    temp_goals = 0
    for n in range(len(df_temp) - 1):
        temp_goals = temp_goals + 1
    df_temp2 = pd.DataFrame({'player':[temp_player],
                            'goals':[temp_goals]})
    player_goals = pd.concat([player_goals, df_temp2], ignore_index=True)

player_goals = player_goals.sort_values('goals')
player_goals = player_goals.reset_index()

games_tag = []
for i in range(len(df)):
    temp_game = df.iloc[i,0]
    if temp_game not in games_tag:
        games_tag.append(temp_game)

df_results = pd.DataFrame(columns = ['game_tag', 'team1', 'team2', 'score'])

count = 0
for games in games_tag:
    temp_team1 = ''
    temp_team2 = ''
    temp_score2 = ''
    df_temp = df[df['id_odsp'] == games]
    if df_temp.iloc[0,7] == 1:
        temp_team1 = df_temp.iloc[0,8]
        temp_team2 = df_temp.iloc[0,9]
    else:
        temp_team1 = df_temp.iloc[0,9]
        temp_team2 = df_temp.iloc[0,8]
    temp_goals = []
    for n in range(len(df_temp)):
        if 'Goal' in df_temp.iloc[n,4]:
            temp_goals.append(df_temp.iloc[n,4])
    x = len(temp_goals) - 1
    temp_score = []
    if x >= 0:
        score_words = temp_goals[x].translate(str.maketrans('','',string.punctuation)).split()
        for word in score_words:
            verifier = True
            if word != '04':
                try:
                    int(word)
                except ValueError:
                    verifier = False
                if verifier:
                    temp_score.append(word)
    if len(temp_score) > 0:
        temp_score2 = str(temp_score[0]) + ' - ' + str(temp_score[1])
    else:
        temp_score2 = '0-0'
    df_temp2 = pd.DataFrame({'game_tag':[games],
                            'team1':[temp_team1],
                            'team2':[temp_team2],
                            'score':[temp_score2]})
    df_results = pd.concat([df_results,df_temp2], ignore_index = False)
    count = count + 1

df_results_no_draws = df_results[df_results['score'] != '0-0']

def player_goal_nos(player_input):
    goals = 0
    df_temp = player_goals[player_goals['player'].str.contains(player_input.lower(), na=False)]
    goals = df_temp.iloc[0,2]

    return goals

def player_score(player_input):
    position = player_goals[player_goals['player'].str.contains(player_input.lower(), na=False)].index
    position = position[0]
    score = position/len(player_goals)
    score = '%.2f' % score

    return score

def player_goal_shot(player_input, data_dict=data_dict):
    df_temp = df_goal[df_goal['player'].str.contains(player_input.lower(), na=False)]
    player_input_goals = df_temp[['shot_place']]
    player_input_goals.dropna(inplace=True)
    x = [data_dict['shot_place'][int(i-1)] for i in player_input_goals['shot_place'].tolist()]
    player_input_goals = pd.DataFrame({'shot_place':x})
    
    return str(max(x))

def player_goal_time(player_input):
    df_temp = df_goal[df_goal['player'].str.contains(player_input.lower(), na=False)]
    player_input_goals = df_temp[['time']]
    player_input_goals.dropna()
    player_input_goals = [int(i) for i in player_input_goals['time']]
    average = sum(player_input_goals)/len(player_input_goals)
    
    return str(average)

def fouls_in_match(team1, team2):
    teams = [team1, team2]
    df_temp = df[df['event_team'].isin(teams)]
    df_temp = df_temp[df_temp['opponent'].isin(teams)]
    tag = df_temp.iloc[0,0]
    df_temp = df_temp[df_temp['id_odsp'] == tag]
    fouls = 0

    for n in range(len(df_temp)):
        if df.iloc[n,5] == 3:
            fouls = fouls + 1
    
    return str(fouls)

def subs_in_match(team1, team2):
    teams = [team1, team2]
    df_temp = df[df['event_team'].isin(teams)]
    df_temp = df_temp[df_temp['opponent'].isin(teams)]
    tag = df_temp.iloc[0,0]
    df_temp = df_temp[df_temp['id_odsp'] == tag]
    substitutions = 0

    for n in range(len(df_temp)):
        if df.iloc[n,5] == 7:
            substitutions = substitutions + 1
    
    return str(substitutions)

def match_results(team1, team2):
    teams = [team1, team2]
    df_temp = df_results[df_results['team1'].isin(teams)]
    df_temp = df_temp[df_temp['team2'].isin(teams)]
    result = df_temp.iloc[0,3]

    return str(result)

def player_team(player_input):
    df_temp = df[df['player'].str.contains(player_input.lower(), na=False)]
    team = df_temp['event_team'].mode().tolist()
    team = ''.join(team)
    
    return str(team)