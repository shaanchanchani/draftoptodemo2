import streamlit as st 
import pandas as pd
import re
import numpy as np
import pickle
import requests
from pathlib import Path

def get_espn_ids(player_dict):
    id_dict = {}
    player_list = [x[0] for x in player_dict.keys()]
    url = 'https://sports.core.api.espn.com/v3/sports/football/nfl/athletes?limit=20000'
    response = requests.get(url)
    data = response.json()
    
    if 'items' in data:
        for item in data['items']:
            if item['fullName'] in player_list:
                id_dict[item['fullName']] = 'https://a.espncdn.com/i/headshots/nfl/players/full/' + str(item['id']) + '.png'
    special_cases = {'Patrick Mahomes II' : 'https://a.espncdn.com/i/headshots/nfl/players/full/3139477.png',
    'DJ Chark Jr.' : 'https://a.espncdn.com/i/headshots/nfl/players/full/3115394.png',
    'Tank Dell' : 'https://a.espncdn.com/i/headshots/nfl/players/full/4366031.png',
    'Ronald Jones II' : 'https://a.espncdn.com/i/headshots/nfl/players/full/3912550.png'}

    id_dict = {**id_dict, **special_cases}

    dst_list = []

    for player in player_list:
        if player not in id_dict.keys():
            dst_list.append(player)

    url = 'http://site.api.espn.com/apis/site/v2/sports/football/nfl/teams'
    response = requests.get(url)
    data = response.json()

    team_data = {}
    dst_dict = {}

    sports = data["sports"]
    for sport in sports:
        leagues = sport["leagues"]
        for league in leagues:
            teams = league["teams"]
            for team in teams:
                team_name = team["team"]["displayName"]
                logo_link = team["team"]["logos"][0]["href"]
                team_data[team_name] = logo_link

    for key,val in team_data.items():
        if key in dst_list:
            dst_dict[key] = val 
        
    id_dict = {**id_dict, **dst_dict}
    return id_dict

def initialize_teams(num_teams):
    for i in range(1,num_teams+1):
        positions = ['QB', 'RB1', 'RB2', 'WR1', 'WR2', 'TE', 'FLEX', 'DST', 'K', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6']
        if f'{i}' not in st.session_state: st.session_state[f'{i}'] = {pos: None for pos in positions}

# def handle_num_teams():
#     if st.session_state.num_teams_key:
#         st.session_state['num_teams'] = st.session_state.num_teams_key
#         st.session_state['user_first_pick'] = 0 

def handle_league_format():
    if st.session_state.lf_key:
        st.session_state['num_teams'] = 12
        st.session_state['user_first_pick'] = 0 

def handle_user_first_pick():
    st.session_state['user_first_pick'] = st.session_state.ufp_key

def handle_make_pick():
    if st.session_state.pick_key: # If make pick button pressed
        st.session_state.pick_num = st.session_state.pick_num + 1 #Increment overall pick number
        #Moves picked player to the roster of the current team picking 
        st.session_state[st.session_state.current_team_picking] = assign_player(st.session_state[st.session_state.current_team_picking], st.session_state.pick_sel_key, st.session_state.df)
        #Removes picked player from dataframe
        st.session_state.df = st.session_state.df[st.session_state.df['Player'] != st.session_state.pick_sel_key]

def assign_player(team, player, df):
    position = df.loc[df['Player'] == player, 'POS'].values[0]
    if position == 'QB' and team['QB'] is None:
        team['QB'] = player
    elif position == 'DST' and team['DST'] is None:
        team['DST'] = player
    elif position == 'K' and team['K'] is None:
        team['K'] = player
    elif position == 'RB':
        if team['RB1'] is None:
            team['RB1'] = player
        elif team['RB2'] is None:
            team['RB2'] = player
        elif team['FLEX'] is None:
            team['FLEX'] = player
        else:
            for i in range(1, 8):
                if team[f'B{i}'] is None:
                    team[f'B{i}'] = player
                    break
    elif position == 'WR':
        if team['WR1'] is None:
            team['WR1'] = player
        elif team['WR2'] is None:
            team['WR2'] = player
        elif team['FLEX'] is None:
            team['FLEX'] = player
    elif position == 'TE' and team['TE'] is None:
        team['TE'] = player
    else:
        for i in range(1, 7):
            if team[f'B{i}'] is None:
                team[f'B{i}'] = player
                break
    return team

def create_pick_order():
    pick_order = (list(range(1,(st.session_state.num_teams+1))) + list(range(st.session_state.num_teams,0,-1)))*15
    pick_order = pick_order[:int(len(pick_order)/2)]
    return pick_order

def is_starting_position(position, team):
    if position == 'QB' and team['QB'] is None:
        return True
    elif position == 'RB' and (team['RB1'] is None or team['RB2'] is None or team['FLEX'] is None):
        return True
    elif position == 'WR' and (team['WR1'] is None or team['WR2'] is None or team['FLEX'] is None):
        return True
    elif position == 'TE' and team['TE'] is None:
        return True
    else:
        return False

def teams_need_position(pos, teams_to_check):
    count = 0 
    for i in teams_to_check:
        if pos == 'RB':
            if st.session_state[f'{i}']['RB1'] is None or st.session_state[f'{i}']['RB2'] is None or (st.session_state[f'{i}']['FLEX'] is None and not st.session_state[f'{i}']['WR1'] and not st.session_state[f'{i}']['WR2']):
                count += 1
        elif pos == 'WR':
            if st.session_state[f'{i}']['WR1'] is None or st.session_state[f'{i}']['WR2'] is None or (st.session_state[f'{i}']['FLEX'] is None and not st.session_state[f'{i}']['RB1'] and not st.session_state[f'{i}']['RB2']):
                count += 1
        else:  # For 'QB' and 'TE'
            if st.session_state[f'{i}'][pos] is None:
                count += 1
    return count

def get_teams_between_picks(pick_order):
    value = pick_order[st.session_state.pick_num]
    arr_slice = []
    for i in range(st.session_state.pick_num, len(pick_order)):
        if pick_order[i] == value:
            return arr_slice
        arr_slice.append(pick_order[i])
    
    return []

def is_starting_position(position, team):
    if position == 'QB' and team['QB'] is None:
        return True
    elif position == 'RB' and (team['RB1'] is None or team['RB2'] is None or team['FLEX'] is None):
        return True
    elif position == 'WR' and (team['WR1'] is None or team['WR2'] is None or team['FLEX'] is None):
        return True
    elif position == 'TE' and team['TE'] is None:
        return True
    else:
        return False

def get_remaining_players_repr(player_df,current_pick_num):
    player_df = player_df.sort_values('ADP')

    positions = ['QB', 'RB', 'WR', 'TE', 'DST', 'K']

    remaining_repr = {}
    for pos in positions:
        pos_players = player_df[player_df['POS'] == pos]
        pos_count = len(pos_players)
        pos_adp_values = pos_players['ADP'].nsmallest(3).tolist()


        # If less than 3 players, pad w/ default_adp (max + 10)
        while len(pos_adp_values) < 3:
            pos_adp_values.append(400)

        remaining_repr[pos] = [pos_count] + pos_adp_values

    ret_df = pd.DataFrame(remaining_repr).T
    return ret_df.values

def get_team_roster_repr(team_dict):
    roster_repr = []
    if team_dict['QB'] is not None:
        roster_repr.append(1)
    else:
        roster_repr.append(0)
    
    if team_dict['RB1'] is not None and team_dict['RB2'] is not None :
        roster_repr.append(1)
    else:
        roster_repr.append(0)
    
    if team_dict['WR1'] is not None and team_dict['WR2'] is not None :
        roster_repr.append(1)
    else:
        roster_repr.append(0)
    
    if team_dict['TE'] is not None:
        roster_repr.append(1)
    else:
        roster_repr.append(0)
    
    if team_dict['FLEX'] is not None:
        roster_repr.append(1)
    else:
        roster_repr.append(0)

    if team_dict['DST'] is not None:
        roster_repr.append(1)
    else:
        roster_repr.append(0)

    if team_dict['K'] is not None:
        roster_repr.append(1)
    else:
        roster_repr.append(0)

    if (team[f'B{i}'] is None for i in range(1,7)):
        roster_repr.append(0)
    else:
        roster_repr.append(1)

    return np.array(roster_repr)

#Forms a single state representation based on the two representations 
def get_state_representation(player_df, current_pick_num, team_dict):
    team_roster_repr = get_team_roster_repr(team_dict)
    remaining_players_repr = get_remaining_players_repr(player_df,current_pick_num)
    state_repr = np.concatenate([team_roster_repr, remaining_players_repr], axis=None)
    return state_repr

def get_model_predictions(model, encoder, input_vector):
    probabilities = model.predict_proba(input_vector.reshape(1, -1)) # predict_proba expects 2D array
    top_two = np.argsort(probabilities, axis=1)[0, -2:] # Get top two
    top_two_probs = probabilities[0, top_two] # Get their probabilities
    class_labels = encoder.inverse_transform(top_two) # Convert back to original class labels
    return {class_labels[i]: top_two_probs[i] for i in range(2)}

@st.cache_resource
def get_smart_model():
    path =  Path(__file__).parent / 'smart_model.pkl'
    with open(path, 'rb') as file:
        smart_model = pickle.load(file)
    return smart_model

@st.cache_resource
def get_avg_model():
    path =  Path(__file__).parent / 'avg_model.pkl'
    with open(path, 'rb') as file:
        avg_model = pickle.load(file)
    return avg_model

@st.cache_resource
def get_encoder():
    path =  Path(__file__).parent / 'encoder.pkl'
    with open(path, 'rb') as file:
        encoder = pickle.load(file)
    return encoder

@st.cache_resource
def get_player_dict():
    path =  Path(__file__).parent / 'player_dict.pkl'
    with open(path, 'rb') as file:
        player_dict = pickle.load(file)
    return player_dict

def draft():
    initialize_teams(st.session_state['num_teams'])

    draft_board_column, team_info_column = st.columns([3, 1])  # adjust the numbers to adjust column width

    pick_order = create_pick_order() #Initialize snaking pick order 
    if(st.session_state.pick_num > (st.session_state.num_teams*15)):
        st.session_state.draft_finished = True
    else:
        st.session_state.current_team_picking = pick_order[st.session_state.pick_num - 1] # -1 bc python is index 0
        if st.session_state.current_team_picking == 0: st.session_state.current_team_picking = 1 

        with draft_board_column:
            undrafted_player_list = st.session_state.df['Player']
            selected_player = st.selectbox(f'With pick number {st.session_state.pick_num} in the draft, Team {st.session_state.current_team_picking} selected...', undrafted_player_list, key = 'pick_sel_key')
            st.button('Make pick', on_click = handle_make_pick, key = 'pick_key')

            if (st.session_state.current_team_picking == st.session_state.user_first_pick and st.session_state.pick_num > st.session_state.num_teams):
                st.header("You're on the board!")
                current_draft_board = st.session_state.df.copy(deep=True)

                # scores_df = calculate_scores(current_draft_board, get_teams_between_picks(pick_order))
                # top_picks = scores_df.sort_values(by='Score', ascending=False).head(5)

                # for _,row in top_picks.iterrows():
                #     st.write(f"{row['Player']} ({row['POS']}) - Score: {row['Score']} {'*' if is_starting_position(row['POS'], st.session_state[st.session_state.current_team_picking]) else ''}")
                state_repr = get_state_representation(current_draft_board, st.session_state.pick_num, st.session_state[st.session_state.current_team_picking])
                smart_model = get_smart_model()
                avg_model = get_avg_model()
                encoder = get_encoder()

                smart_model_predictions = get_model_predictions(smart_model, encoder, state_repr)
                avg_model_predictions = get_model_predictions(avg_model, encoder, state_repr)

                smart_model_predictions = dict(sorted(smart_model_predictions.items(), key=lambda item: item[1], reverse = True))
                avg_model_predictions = dict(sorted(avg_model_predictions.items(), key=lambda item: item[1], reverse = True))

                smart_col, avg_col = st.columns([1,1])

                with smart_col:
                    st.write("Smart Drafter Model")
                    for i, (label, value) in enumerate(smart_model_predictions.items()):
                        val = value*100
                        if i == 0:
                            st.metric(label=label, value=f'{val:.2f}%')
                        elif i == 1:
                            st.metric(label=label, value=f'{val:.2f}%')
                    # smart_chart_data = pd.DataFrame(list(smart_model_predictions.values()), index=list(smart_model_predictions.keys()), columns=['Probability'])
                    # fig1 = go.Figure(data=[go.Bar(x=smart_chart_data.index, y=smart_chart_data['Probability'])])
                    # fig1.update_yaxes(range=[0, 1], autosize = False, width = 100, height = 100)
                    # st.plotly_chart(fig1)
                with avg_col:
                    st.write("Average Drafter Model")
                    for i, (label, value) in enumerate(avg_model_predictions.items()):
                        val = value*100
                        if i == 0:
                            st.metric(label=label, value=f'{val:.2f}%')
                        elif i == 1:
                            st.metric(label=label, value=f'{val:.2f}%')
                    # avg_chart_data =  pd.DataFrame(list(avg_model_predictions.values()), index=list(avg_model_predictions.keys()), columns=['Probability'])
                    # fig2 = go.Figure(data=[go.Bar(x=avg_chart_data.index, y=avg_chart_data['Probability'])])
                    # fig2.update_yaxes(range=[0, 1], autosize = False, width = 100, height = 100)
                    # st.plotly_chart(fig2)
            elif (st.session_state.current_team_picking == st.session_state.user_first_pick and st.session_state.pick_num <= st.session_state.num_teams):
                st.header("You're on the board!")
                st.write("Make your first round pick. Model suggestions will display for your remaining picks.")
            

            st.header("Draft Board")
            st.dataframe(st.session_state.df.sort_values('ADP'), use_container_width = True, column_config = {' ' : st.column_config.ImageColumn()}, hide_index=True)

        with team_info_column:
            with st.expander("Your Roster", expanded = False):
                for key, value in st.session_state[str(st.session_state['user_first_pick'])].items():
                    if value is None:
                        st.write(key)
                    else:
                        st.write(value)

            with st.expander("View another team's roster", expanded = False):
                team_to_display = st.selectbox('Select team to view', [f'Team {i}' for i in range(1, st.session_state['num_teams'] + 1) if i != st.session_state['user_first_pick']], label_visibility='hidden')

                teamID = int(re.sub(r'\D', '', team_to_display))

                for key, value in st.session_state[f'{teamID}'].items():
                    if value is None:
                        st.write(key)
                    else:
                        st.write(value)

def main():
    APP_TITLE = 'Fantasy Football Snake Draft Optimizer'
    st.set_page_config(APP_TITLE, layout = 'wide')
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    if 'player_dict' not in st.session_state:
        # player_dict = np.load('player_dict.npy',allow_pickle=True)
        player_dict = get_player_dict()
        st.session_state.player_dict = dict(player_dict.items())

    if 'id_dict' not in st.session_state:
        st.session_state.id_dict = get_espn_ids(st.session_state.player_dict)

    #Initialize and cleans dataframe
    if 'df' not in st.session_state:
        player_df = pd.DataFrame(list(st.session_state.player_dict.keys()), columns=['Player', 'Team', 'POS'])
        player_df['ADP'] = list(st.session_state.player_dict.values())
        player_df.insert(0, ' ', 0)
        player_df[' '] = player_df['Player'].map(st.session_state.id_dict) 
        st.session_state.df = player_df
    
    #Draft Control Variables
    if 'num_teams' not in st.session_state: st.session_state['num_teams'] = 0
    if 'user_first_pick' not in st.session_state: st.session_state['user_first_pick'] = -1
    if 'current_team_picking' not in st.session_state: st.session_state['current_team_picking'] = 1 
    if 'draft_started' not in st.session_state: st.session_state['draft_started'] = False
    if 'pick_num' not in st.session_state: st.session_state['pick_num'] = 1
    if 'draft_finished' not in st.session_state: st.session_state['draft_finished'] = False

    #If the number of teams hasn't been specified yet (still is 0), prompt user to enter value
    if st.session_state['num_teams'] == 0:
        padcol1,center_col,padcol2 = st.columns([1, 1, 1])  #Padding number input widget makes it look better 
        # center_col.number_input("How many teams are in your draft?", on_change = handle_num_teams, key = 'num_teams_key', step = 1, value = 0)
        center_col.write("Select a league format:")
        center_col.button('12 Team PPR', on_click = handle_league_format, key = 'lf_key')
    
    #If the slot the user is picking from hasn't been specified yet (still is 0), prompt user to enter value
    if st.session_state['user_first_pick'] == 0:
        padcol1, center_col,padcol2 = st.columns([1, 1, 1])  
        center_col.number_input("What slot are you drafting from?", key = 'ufp_key', step = 1, value = 1, min_value = 1, max_value = st.session_state.num_teams)
        center_col.button('Confirm Draft Slot', on_click = handle_user_first_pick)

    #If both the previous values are set, start draft
    if st.session_state['num_teams'] != 0 and st.session_state['user_first_pick'] != 0: st.session_state['draft_started'] = True

    if st.session_state['draft_started']: draft()

    if st.session_state['draft_started'] and st.session_state['draft_finished']: st.header('Draft Finished')


if __name__ == "__main__":
    main()
