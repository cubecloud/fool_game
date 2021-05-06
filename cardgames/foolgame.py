#  Copyright (c) 2021. Oleg Novokshonov
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import random
import numpy as np
import pickle as pkl
import collections
import copy
import time
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import layers

# from tensorflow.keras.layers import Dense, Flatten, Input, Lambda, Conv2D, MaxPooling2D, Reshape, Multiply
# from tensorflow.keras.layers import BatchNormalization
# from tensorflow.keras.optimizers import RMSprop, Adam, SGD, RMSprop

__version__ = "0.01.85"

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'next_state'])


def q_model_conv(in_shape=(37, 25,), num_actions=37):
    # initializer = tf.keras.initializers.RandomUniform(minval=0., maxval=0.05)
    initializer = tf.keras.initializers.GlorotUniform()
    inputs = layers.Input(shape=in_shape)
    # Convolutions on the player deck state
    layer1 = layers.Conv1D(32, 8, strides=4, activation="relu", kernel_initializer=initializer)(inputs)
    layer2 = layers.Conv1D(64, 4, strides=2, activation="relu", kernel_initializer=initializer)(layer1)
    layer3 = layers.Conv1D(64, 3, strides=1, activation="relu", kernel_initializer=initializer)(layer2)

    layer4 = layers.Flatten()(layer3)

    layer5 = layers.Dense(512, activation="relu", kernel_initializer=initializer)(layer4)
    action = layers.Dense(num_actions, activation="linear", kernel_initializer=initializer)(layer5)
    return tensorflow.keras.Model(inputs=inputs, outputs=action)


def q_model_dense(in_shape=(37, 25,), num_actions=37):
    # initializer = tf.keras.initializers.RandomUniform(minval=0., maxval=0.05)
    initializer = tf.keras.initializers.GlorotUniform()
    inputs = layers.Input(shape=in_shape)
    layer0 = layers.Flatten()(inputs)
    layer1 = layers.Dense(128, activation="relu", kernel_initializer=initializer)(layer0)
    layer2 = layers.Dense(256, activation="relu", kernel_initializer=initializer)(layer1)
    action = layers.Dense(num_actions, activation="linear", kernel_initializer=initializer)(layer2)
    return tensorflow.keras.Model(inputs=inputs, outputs=action)


class ExperienceReplay:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = collections.deque(maxlen=capacity)
        pass

    def __len__(self):
        return len(self.buffer)

    def get_length(self):
        return self.__len__()

    def append(self, experience):
        self.buffer.append(experience)
        pass

    def appendleft(self, experience):
        self.buffer.appendleft(experience)
        pass

    def extend(self, experience):
        self.buffer.extend(experience)
        pass

    def show(self):
        buffer_length = self.get_length()
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in range(buffer_length)])
        return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),
                np.array(dones, dtype=np.uint8), np.array(next_states))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),
                np.array(dones, dtype=np.uint8), np.array(next_states))

    def save(self, file_path, buffer_length=10000):
        len_buffer = len(self.buffer)
        with open(file_path, "wb") as f:
            print('Save exp buffer...')
            if not (self.capacity is None) \
                    and (len_buffer < self.capacity) \
                    and (len_buffer < buffer_length):
                buffer_length = len_buffer
            else:
                buffer_length = len_buffer
            states, actions, rewards, dones, next_states = \
                zip(*[self.buffer[idx] for idx in range(len(self.buffer) - buffer_length, len(self.buffer))])
            pkl.dump([states, actions, rewards, dones, next_states], f)
            del [states, actions, rewards, dones, next_states]
            pass

    def load(self, file_path):
        with open(file_path, "rb") as f:
            print('Loading exp buffer...')
            # self.buffer = pkl.load(f)
            states, actions, rewards, dones, next_states = pkl.load(f)
            for state, action, reward, done, next_state in zip(states, actions, rewards, dones, next_states):
                exp = Experience(state, action, reward, done, next_state)
                self.buffer.append(exp)
            del [states, actions, rewards, dones, next_states]
        pass


class ColorText:
    purple = '\033[95m'
    cyan = '\033[96m'
    darkcyan = '\033[36m'
    blue = '\033[94m'
    green = '\033[92m'
    yellow = '\033[93m'
    red = '\033[91m'
    bold = '\033[1m'
    underline = '\033[4m'
    end = '\033[0m'
    red_on_white = '\033[1;31;40m'
    white_on_gray = '\033[1;30;47m'
    gray_on_white = '\033[1;90;40m'


class Deck:
    """
    Вектор np.array (
        [0] - Масть [1..4] (включительно)
        [1] - ранг [1..9] (включительно)
        Статус:
        [2] принадлежность игроку [1..6] (включительно)
            - 0 - лежит в колоде (status - unknown)
            - 1 - карта находится у игрока с номером 1
            - 2 - карта находится у игрока с номером 2
            - 3 - карта находится у игрока с номером 3
            - 4 - карта находится у игрока с номером 4
            - 5 - карта находится у игрока с номером 5
            - 6 - карта находится у игрока с номером 6
        [3] Статус карты
            - 0 - лежит в колоде (status - unknown)
            - 1 - атакующая лежит на столе
            - 2 - защитная лежит на столе
            - 3 - в руке игрока (если мы видели карту) то запоминаем
            - 4 - объявленный козырь (лежит в колоде последним)
        [4] сброс (graveyard)
            - 0 - не в сбросе
            - 1 - в сбросе
        [5] Номер раунда
            - 0 если карта в колоде или у игрока в руке
            - N если карта выложена на стол на раунде N
        [6] - Вес карты
    """

    def __init__(self):
        self.player_deck = {
            # Это инициализация масти пики - 9 карт от 6 пик до туза пик
            1: [1, 1, 0, 0, 0, 0, 1], 2: [1, 2, 0, 0, 0, 0, 2], 3: [1, 3, 0, 0, 0, 0, 3],
            4: [1, 4, 0, 0, 0, 0, 4], 5: [1, 5, 0, 0, 0, 0, 5], 6: [1, 6, 0, 0, 0, 0, 6],
            7: [1, 7, 0, 0, 0, 0, 7], 8: [1, 8, 0, 0, 0, 0, 8], 9: [1, 9, 0, 0, 0, 0, 9],
            # Это инициализация масти крести - 9 карт от 6 крестей до туза крестей
            10: [2, 1, 0, 0, 0, 0, 1], 11: [2, 2, 0, 0, 0, 0, 2], 12: [2, 3, 0, 0, 0, 0, 3],
            13: [2, 4, 0, 0, 0, 0, 4], 14: [2, 5, 0, 0, 0, 0, 5], 15: [2, 6, 0, 0, 0, 0, 6],
            16: [2, 7, 0, 0, 0, 0, 7], 17: [2, 8, 0, 0, 0, 0, 8], 18: [2, 9, 0, 0, 0, 0, 9],
            # Это инициализация масти буби - 9 карт от 6 бубей до туза бубей
            19: [3, 1, 0, 0, 0, 0, 1], 20: [3, 2, 0, 0, 0, 0, 2], 21: [3, 3, 0, 0, 0, 0, 3],
            22: [3, 4, 0, 0, 0, 0, 4], 23: [3, 5, 0, 0, 0, 0, 5], 24: [3, 6, 0, 0, 0, 0, 6],
            25: [3, 7, 0, 0, 0, 0, 7], 26: [3, 8, 0, 0, 0, 0, 8], 27: [3, 9, 0, 0, 0, 0, 9],
            # Это инициализация масти черви - 9 карт от 6 червей до туза червей
            28: [4, 1, 0, 0, 0, 0, 1], 29: [4, 2, 0, 0, 0, 0, 2], 30: [4, 3, 0, 0, 0, 0, 3],
            31: [4, 4, 0, 0, 0, 0, 4], 32: [4, 5, 0, 0, 0, 0, 5], 33: [4, 6, 0, 0, 0, 0, 6],
            34: [4, 7, 0, 0, 0, 0, 7], 35: [4, 8, 0, 0, 0, 0, 8], 36: [4, 9, 0, 0, 0, 0, 9],
        }

        self.suit_range = {'П': (1, 10), 'К': (10, 19), 'Б': (19, 28), 'Ч': (28, 37)}
        self.rank_names = {1: '6', 2: '7', 3: '8', 4: '9', 5: '10', 6: 'В', 7: 'Д', 8: 'К', 9: 'Т'}
        self.suit_chars = {1: 'П', 2: 'К', 3: 'Б', 4: 'Ч'}
        self.suits_names = {1: "Пики", 2: "Крести", 3: "Бубны", 4: "Черви"}
        self.suits_icons = {'П': '\u2660', 'К': '\u2663', 'Б': '\u2666', 'Ч': '\u2665'}
        self.debug_verbose = 1

    pass

    def change_card_status(self, index, status):
        self.player_deck[index][2:6] = status
        pass

    def get_cardinfo(self, index):
        return self.player_deck[index]

    def get_current_status(self, index):
        return self.player_deck[index][2:6]

    def change_card_weight(self, index, new_weight):
        self.player_deck[index][6] = new_weight
        pass

    def get_card_weight(self, index):
        return self.player_deck[index][6]

    def add_card_weight(self, index, add_weight):
        return self.change_card_weight(index, self.get_card_weight(index) + add_weight)

    def what_suit(self, index):
        return self.suit_chars[self.player_deck[index][0]]

    def what_rank(self, index):
        return self.player_deck[index][1]

    def add_weight_2suit(self, suit_char, add_weight):
        for index in range(self.suit_range[suit_char][0], self.suit_range[suit_char][1]):
            self.add_card_weight(index, add_weight)
        pass

    def show_card(self, index):
        suit = self.what_suit(index)
        # Пока не удалось решить проблему с выводом цветного текста сразу из переменной словаря
        # Делаем проверку и присваиваем цвет через print(f'{переменная}'
        if (suit == 'П') or (suit == 'К'):
            color = ColorText.gray_on_white
        else:
            color = ColorText.red_on_white
        card_rank = self.rank_names.get(self.what_rank(index))
        output = f'{color}' + card_rank + f'{color}' + str(
            self.suits_icons[self.what_suit(index)][0:]) + f'{ColorText.end}'
        return output

    def show_cards_hor(self, cards_list):
        output = str()
        for card in cards_list:
            output += (str(self.show_card(card)) + ' ')
            # print(output)
        return output

    def show_cards_vert_numbered(self, cards_list):
        cards_on_hand = 1
        for card in cards_list:
            print(f'{cards_on_hand}. ' + self.show_card(card))
            cards_on_hand += 1
        pass


class Player(Deck):
    def __init__(self,
                 player_number,
                 player_type_num,
                 epsilon):
        super().__init__()
        self.player_number = player_number
        self.player_types = {1: 'Human', 2: 'Computer', 3: 'AI'}
        self.player_type = self.player_types[player_type_num]
        ''' random order '''
        self.epsilon = epsilon
        # если человек то запрашиваем имя
        if self.player_type == self.player_types[1]:
            # self.player_name = self.player_types[player_type]
            self.ask_for_name()
        else:
            self.player_name = f'{self.player_types[player_type_num]} №' + str(self.player_number)
        self.player_cards_onhand_list = list()
        self.game_round: int = 0
        self.player_turn: int = 0
        self.game_turn: int = 0
        self.players_number: int = 0
        self.desktop_list = list()
        self.passive_player_pass_flag = False
        self.attack_player_pass_flag = False
        self.action = None
        self.trump_index = None
        self.trump_char: str = ''
        self.trump_range = tuple
        self.turn_state = np.zeros(shape=(37, 21+self.players_number), dtype=np.float32)
        self.zeros_state = np.zeros(shape=(37, 21+self.players_number), dtype=np.float32)
        self.turn_action_idx: int = 0
        self.turn_experience = tuple()
        self.round_experience: list = []
        self.episode_experience: list = []
        self.episode_buffer: list = []
        self.game_reward: float = 0
        '''
        Zero reward - should be added at the end of episode
        float type
        '''
        self.zero_reward: float = 0
        '''
        Zero done flag
        for signal of ending of playing in current episode for this player
        True should be added at the end of episode
        False(0) or True(1)
        '''
        self.zero_done: float = 0
        '''
        Zero action index
        for signal of ending of playing in current episode for this player
        form -1 to 36 (now we are using 0-37 vector w/o -1)
        '''
        self.zero_action_idx: int = 0
        self.converting_players_order = []
        pass

    @staticmethod
    def convert_2ohe(value, max_value, min_value=1):
        ohe = np.zeros((max_value, ), dtype=np.float32)
        ohe[value - min_value] = 1
        return list(ohe)

    def __converting_list(self):
        self.converting_players_order = [player_num for player_num in range(self.player_number, self.players_number+1)]
        for player_num in range(1, self.player_number):
            self.converting_players_order.append(player_num)
        self.converting_players_order.insert(0, 0)
        pass

    def convert_card_property(self, card_property):
        return self.converting_players_order.index(card_property)

    def convert_deck_2state(self) -> np.array:
        """
        Returns:
            state (np.array):   deck dictionary converted to state
        """
        state = []
        card_state: list = []
        if not self.converting_players_order:
            self.__converting_list()
        '''
        Add Zero action_idx to states (pass)
        '''

        state.append(list(card_state))
        state[0] = list(np.zeros(shape=(21+self.players_number), dtype=np.float32))
        for card_value in self.player_deck.values():
            '''
            ohe suits data - 4 suits
            '''
            card_state.extend(self.convert_2ohe(card_value[0], 4))
            '''
            ohe rank of card data (9 cards)
            '''
            card_state.extend(self.convert_2ohe(card_value[1], 9))
            '''
            ohe card as property of player (self.players_number)
            zero is property of deck
            # Normalize card as property of player (self.players_number)
            '''
            card_property_of = self.convert_card_property(card_value[2])
            card_property_of_ohe = self.convert_2ohe(card_property_of, self.players_number+1, min_value=0)
            card_state.extend(card_property_of_ohe)
            # card_state[2] = card_value[2] / self.players_number
            '''
            ohe card status (possible statuses = 4 (0 NOT included))
            # ohe card status (possible statuses = 5 (0 included))
            # Normalize card status (possible statuses = 4 (0 not included))
            '''
            card_state.extend(self.convert_2ohe(card_value[3], 4))
            # card_state[3] = card_value[3] / 4
            '''                    
            Normalize card graveyard status (zero or 1)
            do not need normalization
            '''
            card_state.append(card_value[4])
            # card_state[4] = card_value[4]
            '''
            # 6. remove round number completely
            # 5. Normalized /10
            # 4. Normalized                                      
            # 3. Normalized /100
            # 2. Will not be normalized cos we doesn't know the total rounds 
            # 1. Normalize round number (will be normalized after playing full episode)            
            '''
            # card_state.append(card_value[5]/10)
            # card_state[5] = card_value[5]/10
            # card_state[5] = card_value[5]
            # card_state[5] = card_value[5]/100
            '''                                     
            # Normalize card weight (max card_weight=34)
            Normalize card weight (max card_weight=18)
            '''
            # card_state.extend(self.convert_2ohe(card_value[6], 34, min_value=0))
            # card_state[6] = card_value[6] / 34
            card_state.append(card_value[6] / 18)
            # state.append(copy.deepcopy(card_state))
            state.append(card_state)
            card_state: list = []
        return np.array(state, dtype=np.float32)

    def add_turn_experience(self, action_idx) -> None:
        """
        Prepare and add one turn experience data to one round experience data

        Args:
            action_idx: action card index from player turn

        Returns:
            None:
        """
        '''
        for temporary testing purpose
        if action_idx == -1:
            action_idx = 0
        '''

        self.game_turn += 1
        self.turn_state = self.convert_deck_2state()
        self.turn_experience: tuple = (self.turn_state, action_idx, self.zero_reward, self.zero_done)
        self.round_experience.append(self.turn_experience)
        pass

    def add_round_experience(self):
        """
        Add one round experience data to episode experience

        Returns:
            None
        """
        self.episode_experience.extend(self.round_experience)
        self.round_experience = []
        pass

    @staticmethod
    def convert_action_idx_2ohe(action_idx):
        action_ohe = np.zeros((37, 1), dtype=np.float32)
        action_ohe[action_idx] = 1.0
        return action_ohe

    def add_episode_experience(self, episode_reward):
        reward_decay = 0.99
        temp_reward = 0
        # max_round: int = 0
        next_state = self.zeros_state
        for turn_idx in range(len(self.episode_experience) - 1, -1, -1):
            turn_state, turn_action_idx, turn_reward, turn_done = self.episode_experience[turn_idx]
            # '''
            # Convert action_idx to ohe vector
            # '''
            # turn_action_idx = self.convert_action_idx_2ohe(turn_action_idx)
            if turn_idx == len(self.episode_experience) - 1:
                ''' Removed normalization for rounds '''
                # if np.max(turn_state[:, 5]) > max_round:
                #     max_round = np.max(turn_state[:, 5])
                # temp_reward = copy.deepcopy(episode_reward)
                next_state = copy.deepcopy(turn_state)
                # print(f'Player number: {self.player_number}, turn_idx {turn_idx} last state action_idx: {turn_action_idx}\n')
                # sys.exit()
                # '''
                # Normalize q-ty of rounds in next_state
                # '''
                # next_state[:, 5] = next_state[:, 5] / max_round
                continue
            elif turn_idx == len(self.episode_experience) - 2:
                turn_done = True
                turn_reward = float(episode_reward)
                temp_reward = episode_reward
            #     print(f'Player number: {self.player_number}, turn_idx {turn_idx}, previous last state action_idx: {turn_action_idx}')
            # else:
            #     print(f'Player number: {self.player_number}, turn_idx {turn_idx}, state action_idx: {turn_action_idx}')
            # '''
            # Normalize q-ty of rounds in state
            # '''
            # turn_state[:, 5] = turn_state[:, 5] / max_round
            if turn_reward == 0:
                temp_reward = temp_reward * reward_decay
                turn_reward = temp_reward
            else:
                temp_reward = temp_reward * reward_decay
                turn_reward += temp_reward
            self.episode_buffer.insert(0, (turn_state, turn_action_idx, turn_reward, turn_done, next_state))
            next_state = copy.deepcopy(turn_state)
        pass

    def change_game_round(self, game_round):
        self.game_round = game_round
        pass

    def change_player_turn(self, turn):
        self.player_turn = turn
        pass

    def add_graveyard_status(self, index):
        """
        Change card status to graveyard
        # card status self.player_deck[index][2:6]

        """
        ''' get current status '''
        status = self.get_current_status(index)

        ''' remove card from any player - status[2] '''
        status[0] = 0

        ''' remove card from any action - status[3] '''
        status[1] = 0

        ''' add graveyard status '''
        status[2] = 1

        # round number for graveyard
        status[3] = self.game_round
        self.change_card_status(index, status)
        pass

    def add_attack_status(self, index):
        status = self.get_current_status(index)
        # player number
        status[0] = self.player_number
        # attack status
        status[1] = 1
        # round number for attack
        status[3] = self.game_round
        self.change_card_status(index, status)
        pass

    def add_player_status(self, index):
        status = self.get_current_status(index)
        # player number
        status[0] = self.player_number
        # on hand status
        status[1] = 3
        self.change_card_status(index, status)
        pass

    def add_defending_status(self, index):
        status = self.get_current_status(index)
        # player number
        status[0] = self.player_number
        # defend status
        status[1] = 2
        # round number for attack
        status[3] = self.game_round
        self.change_card_status(index, status)
        pass

    # Возвращает лист возможных ходов (на основе карт в руке)
    def get_validated_attack_list(self):
        rank_list = list()
        attack_list = list()
        for card in self.desktop_list:
            rank_list.append(self.what_rank(card))
        for card in self.player_cards_onhand_list:
            if self.what_rank(card) in rank_list:
                attack_list.append(card)
        # print (self.show_cards_hor(attack_list))
        return attack_list

    def get_validated_defend_list(self, index) -> list:
        """
        Returns list of valid cards for defending

        Args:
            index: attacking card index

        Returns:
            defend_list: list of valid cards for defend from attacking card
        """

        defend_list = list()
        # trumps_list = list()
        suit = self.what_suit(index)
        weight = self.get_card_weight(index)
        for card in self.player_cards_onhand_list:
            if self.what_suit(card) == suit and self.get_card_weight(card) > weight:
                defend_list.append(card)
        if suit == self.trump_char:
            pass
            # for card in self.trumps_from_hand():
            #     if self.get_card_weight(card) > weight:
            #         trumps_list.append(card)
            # defend_list = trumps_list
        else:
            defend_list.extend(self.trumps_from_hand())
        # print (self.show_cards_hor(defend_list))
        return defend_list

    # возвращает из руки карту с наименьшим индексом
    def lowest_from_hand(self):
        return min(self.player_cards_onhand_list)

    # возвращает из руки _козырную_ карту с наименьшим индексом
    def lowest_trump_from_hand(self):
        try:
            temp = min(self.trumps_from_hand())
            return temp
        except ValueError:
            temp = []
            return temp

    # возвращает List из козырных карт в руке
    def trumps_from_hand(self):
        trumps_onhand = []
        for index in self.player_cards_onhand_list:
            if index in self.trump_range:
                trumps_onhand.append(index)
        return trumps_onhand

    def get_card(self, index):
        self.player_cards_onhand_list.append(index)
        self.add_player_status(index)
        pass

    def check_hand_before_round(self) -> int:
        """
        Returns qty of cards player need to get to have "full" hand

        Returns:
            temp (int): qty of cards to have "full" hand
        """

        cards_qty = 6 - (len(self.player_cards_onhand_list))
        if cards_qty < 0:
            cards_qty = 0
        return cards_qty

    #  это ход игрока
    def turn(self, action):
        self.action = action
        if self.player_type == 'Computer' or self.player_type == 'AI':
            result = self.analyze()
        # elif self.player_type == 'AI':
        #     result = self.analyze()
        else:
            if (self.action == 'Attack') and (len(self.desktop_list) > 0):
                if self.attack_player_pass_flag:
                    result = 0
                    return result
                attack_list = self.get_validated_attack_list()
                while True:
                    card_number = self.ask_for_card()
                    if card_number > 0:
                        result = self.player_cards_onhand_list[card_number - 1]
                        if result in attack_list:
                            # print (self.show_card(result))
                            break
                        else:
                            print('Некорректная карта')
                            continue
                    else:
                        result = 0
                        break
                return result
            elif (self.action == 'Attack') and (len(self.desktop_list) == 0):
                card_number = self.ask_for_card()
                if card_number > 0:
                    result = self.player_cards_onhand_list[card_number - 1]
                    # print (self.show_card(result))
                else:
                    result = 0
                return result

            if (self.action == 'Defend') and (len(self.desktop_list) > 0):
                defending_card = self.desktop_list[(len(self.desktop_list)) - 1]
                # print ('Defending', defending_card)
                defend_list = self.get_validated_defend_list(defending_card)
                while True:
                    card_number = self.ask_for_card()
                    if card_number > 0:
                        result = self.player_cards_onhand_list[card_number - 1]
                        if result in defend_list:
                            # print (self.show_card(result))
                            break
                        else:
                            print('Некорректная карта')
                            continue
                    else:
                        result = 0
                        break
                return result

            if (self.action == 'Passive') and (len(self.desktop_list) > 0) and self.attack_player_pass_flag:
                attack_list = self.get_validated_attack_list()
                while True:
                    card_number = self.ask_for_card()
                    if card_number > 0:
                        result = self.player_cards_onhand_list[card_number - 1]
                        if result in attack_list:
                            print(self.show_card(result))
                            break
                        else:
                            print('Некорректная карта')
                            continue
                    else:
                        result = 0
                        break
                return result
            else:
                # (self.action == 'Passive') and (len(self.desktop_list) == 0):
                result = -1
                return result
            # возвращаем действие и индекс карты (или пас/взять)
            # result = -1
        return result

    def low_weight_card(self, c_list):
        if len(c_list) > 0:
            card_weight = self.player_deck.get(c_list[0])[6]
            low_card = 0
            for card in c_list:
                if (self.player_deck.get(card)[6]) <= card_weight:
                    low_card = card
                    card_weight = self.player_deck.get(card)[6]
        else:
            low_card = 0
        return low_card

    def attacking_low_weight(self):
        # если на столе уже что-то есть
        if len(self.desktop_list) > 0:
            attack_list = self.get_validated_attack_list()
            if len(attack_list) > 0:
                result = self.low_weight_card(attack_list)
            else:
                result = 0
        else:
            result = self.low_weight_card(self.player_cards_onhand_list)
        return result

    def attacking_random(self):
        # если на столе уже что-то есть
        if len(self.desktop_list) > 0:
            attack_list = self.get_validated_attack_list()
            if len(attack_list) > 0:
                ''' 
                Add to random choice one more option if we have something on table 
                0 - pass
                '''
                attack_list.append(0)
                result = random.choice(attack_list)
            else:
                result = 0
        else:
            result = random.choice(self.player_cards_onhand_list)
        return result

    def defending_low_weight(self):
        if not self.attack_player_pass_flag or not self.passive_player_pass_flag:
            check_parity = (len(self.desktop_list) + 1) % 2
            if check_parity == 0:
                # Последняя карта в десктоп листе
                attacking_card = self.desktop_list[(len(self.desktop_list)) - 1]
                # print ('Defending', defending_card)
                defend_list = self.get_validated_defend_list(attacking_card)
                if len(defend_list) > 0:
                    result = self.low_weight_card(defend_list)
                else:
                    result = 0
            else:
                result = -1
        else:
            # если мы ждем то пропускаем к следующему игроку (например когда пасует атакующий)
            result = -1
        return result

    def defending_random(self):
        if not self.attack_player_pass_flag or not self.passive_player_pass_flag:
            check_parity = (len(self.desktop_list) + 1) % 2
            if check_parity == 0:
                # Последняя карта в десктоп листе
                attacking_card = self.desktop_list[(len(self.desktop_list)) - 1]
                # print ('Defending', defending_card)
                defend_list = self.get_validated_defend_list(attacking_card)
                if len(defend_list) > 0:
                    ''' 
                    Add to random choice one more option 
                    0 - get cards
                    '''
                    defend_list.append(0)
                    result = random.choice(defend_list)
                else:
                    result = 0
            else:
                result = -1
        else:
            # если мы ждем то пропускаем к следующему игроку (например когда пасует атакующий)
            result = -1
        return result

    # Атака пассивного компьютера/игрока
    def passive_attacking_low_weight(self):
        if (0 < len(self.desktop_list) < 11) and self.attack_player_pass_flag:
            attack_list = self.get_validated_attack_list()
            if len(attack_list) > 0:

                result = self.low_weight_card(attack_list)
            else:
                result = 0
            return result
        else:
            result = -1
            return result

    def passive_attacking_random(self):
        if (0 < len(self.desktop_list) < 11) and self.attack_player_pass_flag:
            attack_list = self.get_validated_attack_list()
            if len(attack_list) > 0:
                ''' 
                Add to random choice one more option if we have something on table 
                0 - pass
                '''
                attack_list.append(0)
                result = random.choice(attack_list)
            else:
                result = 0
            return result
        else:
            result = -1
            return result

    # Возвращает индекс карты или в случае
    def analyze(self):
        if self.action == 'Attack':
            if random.random() > self.epsilon:
                r_index = self.attacking_low_weight()
            else:
                r_index = self.attacking_random()
        elif self.action == 'Defend':
            if random.random() > self.epsilon:
                r_index = self.defending_low_weight()
            else:
                r_index = self.defending_random()
        else:
            """
            self.action == 'Passive':
            """
            if random.random() > self.epsilon:
                r_index = self.passive_attacking_low_weight()
            else:
                r_index = self.passive_attacking_random()
        return r_index

    def set_trump(self, index):
        self.trump_index = index
        self.trump_char = self.what_suit(index)
        status = self.get_current_status(index)
        ''' trump in current game status self.player_deck[index][2:6] '''
        status[1] = 4
        self.change_card_status(index, status)
        self.add_weight_2suit(self.trump_char, 9)
        self.trump_range = range(self.suit_range[self.trump_char][0], self.suit_range[self.trump_char][1])
        pass

    def get_deck(self, deck):
        self.player_deck = copy.deepcopy(deck)
        pass

    def ask_for_name(self):
        print('Игрок', self.player_number, 'введите имя =>')
        while True:
            try:
                self.player_name = str(input())
                break
            except (TypeError, ValueError):
                print("Неправильный ввод")
        pass

    def ask_for_card(self):
        # print(f'Игрок {self.player_name} введите номер карты =>')
        self.show_cards_vert_numbered(self.player_cards_onhand_list)
        print('0. Пас/забрать')
        self.show_trump()

        while True:
            try:
                card_number = int(input(f'Игрок {self.player_name} введите номер карты =>'))
                if card_number > len(self.player_cards_onhand_list):
                    print("Неправильный ввод")
                    continue
                # elif card_number<len(self.player_cards_onhand_list):
                #     break
                break
            except (TypeError, ValueError):
                print("Неправильный ввод")
        # print (card_number)
        return card_number

    def show_trump(self):
        print(f'Козырь: {self.show_card(self.trump_index)}')
        pass


class AIPlayer(Player):
    def __init__(self,
                 player_number: int,
                 player_type_num: int,
                 nnmodel: tf.keras.Model,
                 epsilon: float
                 ):
        """
        Initialization of AI player

        Args:
            player_number (str):        player number
            player_type_num (int):      player type number
            nnmodel (tf.keras.Model):   nnmodel (keras Model)
            epsilon (float):            current epsilon for random action probability

        Returns:
            None
        """
        super().__init__(player_number, player_type_num, epsilon)
        self.nnmodel = nnmodel
        # self.epsilon = epsilon
        self.num_actions = 37
        pass

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon
        pass

    def get_action(self, action_list):
        if np.random.random() < self.epsilon:
            action = random.choice(action_list)
        else:
            # Predict action Q-values
            # From environment state

            state_a = self.convert_deck_2state()
            state_tensor = tf.convert_to_tensor(state_a)
            state_tensor = tf.expand_dims(state_tensor, 0)
            q_values = self.nnmodel(state_tensor, training=False)
            # with np.printoptions(precision=3, suppress=True):
            #     print(q_values.numpy())
            valid_masks = tf.one_hot(action_list, self.num_actions)
            # with np.printoptions(precision=3, suppress=True):
            #     print(masks.numpy())
            valid_q_values = tf.expand_dims(tf.reduce_sum(tf.multiply(q_values, valid_masks), axis=0), 0)
            # with np.printoptions(precision=3, suppress=True):
            #     print(valid_q_values.numpy())
            action = np.argmax(valid_q_values)
            # print(self.action, action_list, action)
            if not (action in action_list):
                action = action_list[0]
        return action

    def attacking(self):
        # если на столе уже что-то есть
        if len(self.desktop_list) > 0:
            attack_list = self.get_validated_attack_list()
            ''' 
            Add to random choice one more option if we have something on table 
            0 - pass
            '''
            attack_list.insert(0, 0)
            if random.random() < self.epsilon:
                if len(attack_list) > 0:
                    action = random.choice(attack_list)
                else:
                    action = 0
            else:
                action = self.get_action(attack_list)
        else:
            if random.random() < self.epsilon:
                action = random.choice(self.player_cards_onhand_list)
            else:
                action = self.get_action(self.player_cards_onhand_list)
        return action

    def defending(self):
        if not self.attack_player_pass_flag or not self.passive_player_pass_flag:
            check_parity = (len(self.desktop_list) + 1) % 2
            if check_parity == 0:
                # Последняя карта в десктоп листе
                attacking_card = self.desktop_list[(len(self.desktop_list)) - 1]
                # print ('Defending', defending_card)
                defend_list = self.get_validated_defend_list(attacking_card)
                if len(defend_list) > 0:
                    ''' 
                    Add to random choice one more option 
                    0 - get cards
                    '''
                    defend_list.insert(0, 0)
                    if random.random() < self.epsilon:
                        action = random.choice(defend_list)
                    else:
                        action = self.get_action(defend_list)
                else:
                    action = 0
            else:
                action = -1
        else:
            # если мы ждем то пропускаем к следующему игроку (например когда пасует атакующий)
            action = -1
        return action

    def passive_attacking(self):
        if (0 < len(self.desktop_list) < 11) and self.attack_player_pass_flag:
            attack_list = self.get_validated_attack_list()
            if len(attack_list) > 0:
                ''' 
                Add to random choice one more option if we have something on table 
                0 - pass
                '''
                attack_list.insert(0, 0)
                if random.random() < self.epsilon:
                    action = random.choice(attack_list)
                else:
                    action = self.get_action(attack_list)
            else:
                action = 0
        else:
            # если мы ждем то пропускаем к следующему игроку (например когда аттакует атакующий)
            action = -1
        return action

    def analyze(self):
        if self.action == 'Attack':
            card_index = self.attacking()
        elif self.action == 'Defend':
            card_index = self.defending()
        else:
            """
            self.action == 'Passive':
            """
            card_index = self.passive_attacking()
        return card_index
    pass


class Table:
    def __init__(self, players_qty):
        # кол-во игроков
        self.winner = 0
        self.looser = 0
        self.dd = Deck()
        self.players_number = players_qty
        self.players_qty = players_qty
        self.playing_deck = self.dd.player_deck
        self.desktop_list: list = []
        self.end_of_deck = False
        self.result = 0
        self.players_numbers_lst: list = []
        self.hidden_deck_index: int = -1
        self.hidden_playing_deck_order: list = []
        self.trump_index: int = -1
        self.player_turn: int = 0
        self.current_player_id: int = 0
        self.game_round: int = 0
        self.action: str = str()
        self.attack_player_empty_hand_flag = False

        self.pl: dict = {}
        for i in range(1, self.players_number + 1):
            self.players_numbers_lst.append(i)

        self.debug_verbose = 1
        self.episode_players_ranks: list = []
        self.start_time = time.time()
        self.time_elapsed = 0
        self.first_discard = True
        self.game_circle = True
        self.verbose = False
        pass

    def print_msg(self, msg: str) -> None:
        if self.verbose:
            print(msg)
        pass

    def current_card_index(self) -> int:
        """
        передаем индекс карты из списка self.hidden_playing_deck_order,
        ссылаясь на индекс верхней карты колоды

        Returns:
            card_index (int):   top card in the deck index
        """
        return self.hidden_playing_deck_order[self.hidden_deck_index]

    def add_2graveyard(self, graveyard_list) -> None:
        """
        Mark the cards from graveyard_list for each players deck with graveyard status

        Args:
            graveyard_list (list):  list of card for graveyard status

        Returns:
            None
        """

        ''' First discard flag set to False '''
        if self.first_discard:
            self.first_discard = False
        for index in graveyard_list:
            for player_number in self.players_numbers_lst:
                # Убрать карту со стола (поменять статус стола и принадлежности на 'Сброс')
                # Добавим карту в свою базу знаний
                self.pl[player_number].add_graveyard_status(index)
        pass

    def add_card_2desktop(self, index, action, action_player_number):
        if action == "Attack":
            action_number = 1
        elif action == "Defend":
            action_number = 2
        else:
            """
            action == "Passive":
            """
            action_number = 1
        self.desktop_list.append(index)
        for player_id in self.players_numbers_lst:
            self.pl[player_id].desktop_list = self.desktop_list
            status = self.pl[player_id].get_current_status(index)
            # player number
            status[0] = action_player_number
            # on hand status
            status[1] = action_number
            self.pl[player_id].change_card_status(index, status)
        pass

    def rem_cards_from_desktop(self):
        self.desktop_list.clear()
        for player_id in self.players_numbers_lst:
            self.pl[player_id].desktop_list = self.desktop_list
        pass

    def show_desktop(self):
        desktop_list_1 = list()
        desktop_list_2 = list()
        # тут проверяем статус карты и делаем 2 списка в зависимости от игрока и режима.
        for i in range(0, len(self.desktop_list), 2):
            desktop_list_1.append(self.desktop_list[i])
        for i in range(1, len(self.desktop_list), 2):
            desktop_list_2.append(self.desktop_list[i])
        # print (desktop_list_1)
        # print (desktop_list_2)
        if len(desktop_list_1) > 0:
            if self.debug_verbose > 1:
                print(desktop_list_1)
            self.print_msg(self.pl[1].show_cards_hor(desktop_list_1))
        else:
            self.print_msg('')
        if len(desktop_list_2) > 0:
            if self.debug_verbose > 1:
                print(desktop_list_2)
            self.print_msg(self.pl[1].show_cards_hor(desktop_list_2))
        else:
            self.print_msg('')
        pass

    def add_card_2player_hand(self, player_number) -> None:
        """
        добавить карту в руку игрока (раздача, забрать
        если не отбился, или из колоды после отбоя)
        Args:
            player_number:  add card to player_number hand

        Returns:
            None
        """
        if self.hidden_deck_index <= 35 and not self.end_of_deck:
            self.pl[player_number].get_card(self.current_card_index())
            # Индекс карты в дек листе меняем на следующую карту
            self.hidden_deck_index += 1
            if self.hidden_deck_index == 36:
                self.hidden_deck_index = 35
                self.end_of_deck = True
        else:
            self.end_of_deck = True
        pass

    def add_cardslist_2player_hand(self, player_number, cards_list):
        for index in cards_list:
            # взять карту из листа в руку
            self.pl[player_number].get_card(index)
            for player_id in self.players_numbers_lst:
                if player_id != player_number:
                    status = self.pl[player_id].get_current_status(index)
                    # player number
                    status[0] = player_number
                    # on hand status
                    status[1] = 3
                    status[3] = self.game_round
                    self.pl[player_id].change_card_status(index, status)
                else:
                    self.pl[player_id].add_player_status(index)
        pass

    # Установить козыря для всех игроков
    # (фактически показать его всем)
    def add_trump_card(self):
        for player_id in self.players_numbers_lst:
            # Поменять статус на 'Козырь')
            # Добавим карту в базу знаний всех игроков
            # Сделать с индексом козыря (в деке по этому номеру лежит последняя козырная карта в колоде)
            self.pl[player_id].set_trump(self.current_card_index())
        self.trump_index = self.current_card_index()
        # перенести в скрытом листе, открытого козыря последним в список, перемешанной деки
        # чтобы он был последним
        self.hidden_playing_deck_order.remove(self.trump_index)
        self.hidden_playing_deck_order.append(self.trump_index)
        # Индекс карты в дек листе НЕ меняем на следующую карту, она сама передвигается на нужное место
        pass

    # Выбор игрока с первым ходом
    # логика: сначалас ищем игрока с наименьшим козырем,
    # если такого нет ищется игрок с наименьшей картой (по рангу масти)
    def first_turn_choice(self):
        # print ("Идет выбор ходящего первым")
        min_card_index: dict = {}
        for player_id in self.players_numbers_lst:
            # print('Игрок',player_id, self.show_cards_list(self.pl[player_id].trumps_from_hand()))
            min_card_index[player_id] = self.pl[player_id].lowest_trump_from_hand()
        # Пробуем операцию мин
        try:
            min_card_player = min(min_card_index.keys(), key=(lambda k: min_card_index[k]))
            # print (f'Минимальная карта у игрока {min_card_player}, это карта',
            # self.show_card(min_card_index[min_card_player]))
            return min_card_player, min_card_index[min_card_player]
        # если есть пустые листы
        except (TypeError, ValueError):
            check_player = 0
            check_card = 0
            for player_id in self.players_numbers_lst:
                # если лист пуст ищем дальше по циклу
                if not min_card_index[player_id]:
                    continue
                # если список не пустой проверяем значение и записываем если больше
                elif min_card_index[player_id] > check_card:
                    check_card = min_card_index[player_id]
                    check_player = player_id
            # если есть игрок с козырем то выводим
            if check_card != 0:
                min_card_player = check_player
            # Игрока с козырем нет - ищем _любую_ самую младшую карту у игроков (пики, крести, бубны, червы)
            else:
                for player_number1 in self.players_numbers_lst:
                    # print('Игрок',player_number1,
                    # self.show_cards_list(self.pl[player_number1].player_cards_onhand_list))
                    min_card_index[player_number1] = self.pl[player_number1].lowest_from_hand()
                min_card_player = (min(min_card_index.items(), key=lambda x: x[1])[0])
        # print(min_card_index)
        # print (f'Минимальная карта у игрока {min_card_player},
        # это карта', self.show_card(min_card_index[min_card_player]))
        return min_card_player, min_card_index[min_card_player]

    def set_nnmodel(self, model):
        self.nnmodel = model
        pass

    # def set_epsilon(self, epsilon):
    #     self.epsilon = epsilon
    #     pass

    # Устанавливаем кол-во игроков
    def set_players(self):
        # Иницианилизируем работу класса игроков и делаем их словарем
        self.print_msg(f'Кол-во игроков: {self.players_number}')
        for player_id in self.players_numbers_lst:
            if player_id == 1:
                # self.pl[1] = Player(1, 1)
                self.pl[1] = Player(1, 2)
            elif player_id == 2:
                # Тип 2 - Computer
                self.pl[player_id] = Player(player_id, 2)
        pass

    # Инициализируем словарь (массив) с деками игроков
    def table_init(self):
        for player_id in self.players_numbers_lst:
            # заносим в словари деки игроков
            self.pl[player_id].get_deck(self.playing_deck)
            self.pl[player_id].players_number = self.players_number
        # устанавливаем индекс карты из колоды на 1
        # это индекс для ###### self.hidden_playing_deck_order ######
        self.hidden_deck_index = 0
        pass

    def shuffle(self):
        """
        Shuffle the deck of cards

        Returns:
            None
        """
        # Это лист индексов колоды карт который отражает фактически колоду
        self.hidden_playing_deck_order = list(self.playing_deck.keys())
        # А теперь перемешанную колоду
        random.shuffle(self.hidden_playing_deck_order)
        pass

    def show_first_turn_card(self):
        player, index = self.first_turn_choice()
        self.player_turn = player
        self.print_msg(f'Ходит игрок №{player} {self.pl[player].player_name}, у него меньшая карта '
                       f'{self.pl[player].show_card(index)}')
        status = self.pl[player].get_current_status(index)
        # player number
        status[0] = player
        # on hand status
        status[1] = 3
        for player_id in self.players_numbers_lst:
            self.pl[player_id].change_card_status(index, status)
            # self.pl[player_number].change_card_status(index, 'P' + str(player))
        pass

    def set_table(self, start_table='new'):
        self.set_players()
        if start_table != 'same':
            self.shuffle()
        self.table_init()
        # print(self.hidden_playing_deck_order)
        ''' раздача карт '''
        for _ in range(6):
            for player_id in self.players_numbers_lst:
                ''' добавляем 1 карту каждому игроку '''
                self.add_card_2player_hand(player_id)

        ''' Setting trump card '''
        if start_table != 'same':
            self.add_trump_card()
        else:
            for player_id in self.players_numbers_lst:
                self.pl[player_id].set_trump(self.hidden_playing_deck_order[-1])
            self.trump_index = self.hidden_playing_deck_order[-1]

        if start_table == 'new' or start_table == 'same':
            self.show_first_turn_card()

        self.game_round = 0
        self.next_round()
        # print(f'Раунд № {self.game_round}')
        # self.pl[1].show_cards_vert_numbered(self.pl[1].player_cards_onhand_list)
        # print('0. Пас/забрать')
        # self.show_trump()
        pass

    # Нужно переставлять ход ДО вызова.
    def one_more_is_out(self, player_number):
        if player_number == self.player_turn:
            self.next_turn()
        self.players_numbers_lst.remove(player_number)
        # уменьшаем кол-во игроков
        self.players_number -= 1
        # переносим переход хода
        # self.player_turn = self.next_player(self.player_turn)
        pass

    def if_player_hand_and_deck_empty(self, player_number):
        if self.end_of_deck and (len(self.pl[player_number].player_cards_onhand_list) == 0):
            if self.winner == 0:
                self.winner = player_number
            return True
        else:
            return False

    def rem_cards_and_change_the_game(self, player_number):
        # если есть карты на десктопе
        # и карт четное кол-во (то-есть был отбой)
        if len(self.desktop_list) > 0 and len(self.desktop_list) % 2 == 0:
            print(f'Карты уходят в сброс', self.pl[player_number].show_cards_hor(self.desktop_list))
            self.add_2graveyard(self.desktop_list)
            # убираем карты с десктопа
            self.rem_cards_from_desktop()
            # Переход кона
            # self.next_round()
            # Переход хода
            # self.next_turn()
        pass

    def check_end_of_game(self) -> bool:
        """
        Checking for winner and looser

        Returns:
            result (bool):  True if game is over and False if it's not
        """
        result = False
        if self.is_this_end_of_game():
            ''' Debug of the card deck array '''
            if self.debug_verbose > 2:
                for player_id in range(1, self.players_qty + 1):
                    print(f'Player number: {player_id}')
                    with np.printoptions(precision=3, suppress=True):
                        print(self.pl[player_id].convert_deck_2state())
            ''' If game is over '''
            result = True
            self.game_circle = False
            self.congratulations()
        return result

    def calc_rank_reward(self):
        """


        Returns:
        """
        rank_rewards_lst = list(np.linspace(1.0, 0.0, num=self.players_qty))
        rank_reward = rank_rewards_lst[len(self.episode_players_ranks)-1]
        return rank_reward

    def is_this_end_of_game(self) -> bool:
        # Проверка окончания игры если какой_либо игрок закончил игру.
        result = False
        players_id_lst = list(self.players_numbers_lst)
        for player_id in players_id_lst:
            if self.if_player_hand_and_deck_empty(player_id):

                if not (player_id in self.episode_players_ranks):
                    self.episode_players_ranks.append(player_id)

                self.print_msg(
                    f'Игрок № {player_id}, заканчивает игру. Остается {self.players_number - 1} игроков')
                self.show_desktop()
                self.add_2graveyard(self.desktop_list)
                self.rem_cards_from_desktop()

                # if self.pl[player_id].player_type == 'AI':
                rank_reward = self.calc_rank_reward()
                '''
                Why zero (0)?
                '''
                self.pl[player_id].add_turn_experience(0)
                self.pl[player_id].game_reward = float(rank_reward)
                self.pl[player_id].add_round_experience()
                self.pl[player_id].add_episode_experience(rank_reward)

                if self.players_number == 2:
                    self.looser = self.next_player(player_id)
                    self.episode_players_ranks.append(self.looser)
                    # if self.pl[self.looser].player_type == 'AI':
                    rank_reward = self.calc_rank_reward()
                    self.pl[self.looser].game_reward = float(rank_reward)
                    self.pl[self.looser].add_round_experience()
                    self.pl[self.looser].add_episode_experience(rank_reward)
                    result = True

                self.one_more_is_out(player_id)

        return result

    def congratulations(self):
        self.time_elapsed = time.time() - self.start_time
        msg = f'==========================================\n' \
              f'Победитель игрок №{self.winner}\nПроигравший игрок №{self.looser}\n' \
              f'Игра закончена за {self.game_round} раундов и {self.time_elapsed:.2f} сек\n' \
              f'=========================================='
        self.print_msg(msg)

        pass

    def show_trump(self):
        self.print_msg(f'Козырь: {self.pl[1].show_card(self.trump_index)}')
        pass

    def show_all_cards(self, player_number) -> None:
        """
        Show all cards on desktop (table)
        Args:
            player_number:  player number
        Returns:
            None
        """
        self.if_human_pause(player_number)
        # Для теста
        # for i in self.players_numbers_lst:
        #     print(f'Игрок {i}', self.pl[i].show_cards_hor(self.pl[i].player_cards_onhand_list))
        self.print_msg(f'Раунд № {self.game_round}')
        self.print_msg(f'В колоде карт: {(35 - self.hidden_deck_index)}')
        if self.debug_verbose > 1:
            self.print_msg(f'Индексы перемешанной колоды:\n{self.hidden_playing_deck_order}')
            for pl_number in self.players_numbers_lst:
                self.print_msg(f'Player number № {pl_number}:{self.pl[pl_number].player_cards_onhand_list}')
        self.print_msg(f'Заход игрока {self.player_turn}, {self.pl[self.player_turn].player_name}')
        self.print_msg(
            f'Отбой игрока {self.next_player(self.player_turn)}, '
            f'{self.pl[self.next_player(self.player_turn)].player_name}')
        self.show_trump()
        # self.pl[1].show_cards_vert_numbered(self.pl[1].player_cards_onhand_list)
        if len(self.desktop_list) > 0:
            self.print_msg(f'На столе')
            self.show_desktop()
        else:
            self.print_msg(f'Стол пуст\n')
        pass

    def next_turn(self):
        self.player_turn = self.next_player(self.player_turn)
        for player_id in self.players_numbers_lst:
            self.pl[player_id].change_player_turn(self.player_turn)
        pass

    def next_player(self, player_number):
        # Индекс должен быть меньше чем в реальный player_number
        index = self.players_numbers_lst.index(player_number)
        if index + 1 > len(self.players_numbers_lst) - 1:
            index = 0
            return self.players_numbers_lst[index]
        else:
            index += 1
            return self.players_numbers_lst[index]
        pass

    def previous_player(self, player_number):
        index = self.players_numbers_lst.index(player_number)
        if index - 1 == -1:
            index = len(self.players_numbers_lst) - 1
            return self.players_numbers_lst[index]
        else:
            index -= 1
            return self.players_numbers_lst[index]
        pass

    def check_attack_player_pass_flag(self):
        """
        Check the flag {pass} from attacking player

        Проверяем есть ли флаг {пасс} у атакующего игрока
        """
        if self.pl[self.player_turn].attack_player_pass_flag:
            return True
        else:
            return False

    def set_attack_player_pass_flag(self, flag):
        """
        Внимание! Устанавливает флаг, что
        атакующий игрок пасует - ВСЕМ игрокам
        """
        for player_id in self.players_numbers_lst:
            self.pl[player_id].attack_player_pass_flag = flag
        pass

    def set_passive_player_pass_flag(self, player_number, flag):
        """
        Внимание! устанавливает флаг только
        текущему игроку (у кого сейчас ход)
        """
        self.pl[player_number].passive_player_pass_flag = flag

    def check_passive_players_pass_flag(self):
        """
        проверяем есть ли флаг пасс у пассивных игроков
        проверка идет у атакующего игрока
        """
        ''' Copying list of playing players '''
        players_numbers_lst_wo_attacking = list(self.players_numbers_lst)
        ''' Remove attacking player from list '''
        players_numbers_lst_wo_attacking.remove(self.player_turn)
        ''' Remove defending player from list '''
        players_numbers_lst_wo_attacking.remove(self.next_player(self.player_turn))

        is_all_players_passed = True
        for player_id in players_numbers_lst_wo_attacking:
            is_all_players_passed = is_all_players_passed and self.pl[player_id].passive_player_pass_flag
        return is_all_players_passed

    def next_round(self):
        self.game_round += 1
        self.set_attack_player_pass_flag(False)
        for player_id in self.players_numbers_lst:

            ''' Add round experience if player AI '''
            # if self.pl[player_id].player_type == 'AI':
            self.pl[player_id].add_round_experience()

            ''' remove passive players flags '''
            self.set_passive_player_pass_flag(player_id, False)

            ''' adding cards from deck '''
            if not self.end_of_deck:
                for _ in range(self.pl[player_id].check_hand_before_round()):
                    self.add_card_2player_hand(player_id)

            ''' Debug of the card deck array '''
            if self.debug_verbose > 2:
                print(f'Player number: {player_id}')
                with np.printoptions(precision=3, suppress=True):
                    print(self.pl[player_id].convert_deck_2state())
                # for key, value in self.pl[player_id].player_deck.items():
                #     print(key, value)

            self.pl[player_id].change_game_round(self.game_round)
            self.pl[player_id].change_player_turn(self.player_turn)
        pass

    def if_human_pause(self, player_number):
        # flag = False
        # if self.end_of_deck:
        #     for player_number in self.players_numbers_lst:
        #         if len(self.pl[player_number].player_cards_onhand_list) < 2:
        #             flag = True
        # if flag:
        #     time.sleep(2)
        #     self.clear_screen()

        # if self.pl[player_number].player_type == 'Computer' or  self.pl[player_number].player_type == 'AI':
        #     time.sleep(3)
        #     self.clear_screen()
        # else:
        #     time.sleep(2)
        #     self.clear_screen()
        pass

    @staticmethod
    def clear_screen() -> None:
        print("\n" * 100)
        pass

    def take_action(self, player_number):
        if player_number == self.player_turn:
            action = 'Attack'
        # если ход + 1 равен наш номер - защищаемся, или наш номер 1-й в списке, а ходит последний в списке
        elif (player_number == self.next_player(self.player_turn)) or \
                (self.player_turn == self.players_numbers_lst[self.players_number - 1] and
                 player_number == self.players_numbers_lst[0]):
            action = 'Defend'
        else:
            action = 'Passive'
        return action

    def current_player_attack_action(self) -> None:
        if self.result > 0:
            self.attack_player_empty_hand_flag = False
            self.add_card_2desktop(self.result, self.action, self.current_player_id)
            self.pl[self.current_player_id].add_attack_status(self.result)
            self.pl[self.current_player_id].player_cards_onhand_list.remove(self.result)

            ''' Save data about turn experience, with action_idx (self.result) '''
            # if self.pl[self.current_player_id].player_type == 'AI':
            self.pl[self.current_player_id].add_turn_experience(self.result)

            # print(f'Ход игрока {player_number}
            # {self.pl[player_number].player_name} - {self.pl[player_number].show_card(result)}')
            self.print_msg(
                f'Ход игрока {self.current_player_id} {self.pl[self.current_player_id].player_name} под игрока '
                f'{self.next_player(self.current_player_id)} '
                f'{self.pl[self.current_player_id].show_card(self.result)}')
            # print (f'Атака игрока {player_number}',self.pl[player_number].show_card(result))
            # print ('Десктоп', self.desktop_list)
            # передача по кругу следующему игроку
            self.current_player_id = self.next_player(self.current_player_id)
            # выставляем флаг, что мы не пасуем
            self.set_attack_player_pass_flag(False)
            # print ('PN',player_number, 'PT',self.player_turn)
            # self.show_all_cards()
            # self.if_human_pause(player_number)
            # if self.if_player_hand_and_deck_empty(player_number):
            #     self.attack_player_empty_hand_flag = True
            return
        elif self.result == 0:
            ''' if only 2 players and the player action is pass (self.result == 0) '''
            if self.players_number == 2:
                # Карты уходят в сброс того что входит
                self.print_msg(
                    f'Игрок {self.current_player_id} {self.pl[self.current_player_id].player_name} пасует, '
                    f'карты уходят в сброс {self.pl[self.current_player_id].show_cards_hor(self.desktop_list)}')
                self.add_2graveyard(self.desktop_list)
                # убираем карты с десктопа
                self.rem_cards_from_desktop()

                ''' Save data about turn experience, with action_idx (self.result) '''
                # if self.pl[self.current_player_id].player_type == 'AI':
                self.pl[self.current_player_id].add_turn_experience(self.result)

                # Переход хода
                self.next_turn()
                if self.check_end_of_game():
                    return
                # Переход кона,
                self.next_round()
                self.current_player_id = int(self.player_turn)
                return
            # Проверяем флаги, что мы уже пасовали
            # и пассивные игроки пасовали и можно отправить карты в сброс
            elif self.players_number > 2 and \
                    self.check_passive_players_pass_flag() and \
                    self.check_attack_player_pass_flag():
                # Карты уходят в сброс
                self.print_msg(f'Карты уходят в сброс '
                               f'{self.pl[self.current_player_id].show_cards_hor(self.desktop_list)}')
                self.add_2graveyard(self.desktop_list)
                # убираем карты с десктопа
                self.rem_cards_from_desktop()

                ''' Save data about turn experience, with action_idx (self.result) '''
                # if self.pl[self.current_player_id].player_type == 'AI':
                self.pl[self.current_player_id].add_turn_experience(self.result)

                # Переход хода
                self.next_turn()
                # если этот игрок
                if self.check_end_of_game():
                    return
                # Переход кона,
                self.next_round()
                # Смена игрока и переход ему хода
                self.current_player_id = int(self.player_turn)
                return
            # если в сброс карты не отправились, то....
            elif self.players_number > 2:
                # мы пасуем, но может сходить следующий игрок.
                # Поэтому мы передаем ход через 1 игрока (отбивающегося)
                self.print_msg(f'Игрок {self.current_player_id} '
                               f'{self.pl[self.current_player_id].player_name} пасует, можно подбрасывать')

                ''' Save data about turn experience, with action_idx (self.result) '''
                # if self.pl[self.current_player_id].player_type == 'AI':
                #     self.pl[self.current_player_id].add_turn_experience(self.result)

                self.current_player_id = self.next_player(self.next_player(self.current_player_id))
                # и выставляем флаг, что мы пасуем
                self.set_attack_player_pass_flag(True)
            return
        pass

    def current_player_defend_action(self) -> None:
        if self.result > 0:
            self.pl[self.current_player_id].add_defending_status(self.result)
            # print(self.pl[player_number].player_cards_onhand_list, result)
            self.pl[self.current_player_id].player_cards_onhand_list.remove(self.result)
            self.add_card_2desktop(self.result, self.action, self.current_player_id)

            ''' Save data about turn experience, with action_idx (self.result) '''
            # if self.pl[self.current_player_id].player_type == 'AI':
            self.pl[self.current_player_id].add_turn_experience(self.result)

            # print(
            #     f'Игрок {player_number}
            #     {self.pl[player_number].player_name} отбивается - {self.pl[player_number].show_card(result)}')
            self.print_msg(
                f'Игрок {self.current_player_id} {self.pl[self.current_player_id].player_name} '
                f'отбивается {self.pl[self.current_player_id].show_card(self.result)}')
            # print ('Десктоп', self.desktop_list)

            # print ('PN',player_number, 'PT',self.player_turn)
            # self.if_human_pause(player_number)
            '''
            если в руке больше нет карт на отбой и в колоде пусто ИЛИ
            если нам не достанется ничего из колоды при раздаче и в руке нет больше карт на отбой
            - мы выходим из игры и нас исключают из списка играющих. Остальные играют дальше
            Переход хода идет на следующего игрока (следующего за отбивающимся)
            '''
            if self.if_player_hand_and_deck_empty(self.current_player_id) or \
                    ((35 - self.hidden_deck_index) < len(self.desktop_list) // 2 and
                     (len(self.pl[self.current_player_id].player_cards_onhand_list) == 0)):
                if self.players_number != 2:
                    self.next_turn()
                self.next_turn()
                if self.check_end_of_game():
                    return
                self.next_round()
                self.current_player_id = int(self.player_turn)
                return
            elif (len(self.desktop_list) == 12) or \
                    (self.first_discard and len(self.desktop_list) == 10):
                self.print_msg(
                    f'Игрок {self.current_player_id} '
                    f'{self.pl[self.current_player_id].player_name} '
                    f'отбивается, {"первый сброс" if self.first_discard else ""} ({int(len(self.desktop_list)/2)} пар),'
                    f' карты уходят в сброс {self.pl[self.current_player_id].show_cards_hor(self.desktop_list)}')

                self.add_2graveyard(self.desktop_list)
                # убираем карты с десктопа
                self.rem_cards_from_desktop()
                # Переход хода
                if self.check_end_of_game():
                    return
                # Переход кона,
                self.next_turn()
                self.next_round()
                # Смена игрока и переход ему хода
                self.current_player_id = int(self.player_turn)
                # self.if_human_pause(player_number)
                return
            # передача по кругу следующему игроку
            self.current_player_id = self.next_player(self.current_player_id)
            return
        elif self.result == 0:
            # Если забираем карты (нет отбоя)
            self.print_msg(
                f'Игрок {self.current_player_id} '
                f'{self.pl[self.current_player_id].player_name} забирает '
                f'{self.pl[self.current_player_id].show_cards_hor(self.desktop_list)}')

            ''' Мы забрали карты со стола '''
            self.add_cardslist_2player_hand(self.current_player_id, self.desktop_list)

            ''' Save data about turn experience, with action_idx (self.result) '''
            # if self.pl[self.current_player_id].player_type == 'AI':
            self.pl[self.current_player_id].add_turn_experience(self.result)

            # проверяем на наличие карт
            # если игроков 2 и пас то, если 2 игрока просто следующий кон,
            # но игрок остается тот-же
            if self.players_number == 2:
                if self.check_end_of_game():
                    return
                # убираем карты с десктопа
                self.rem_cards_from_desktop()
                # Переход кона,
                self.next_round()
                # Перехода хода нет - ход остается у прежнего игрока
                # передаем ход ему обратно
                self.current_player_id = self.player_turn
            elif self.players_number > 2:
                ''' если игроков больше 2-х, переход хода на 2 вперед '''
                self.next_turn()
                self.next_turn()
                if self.check_end_of_game():
                    return
                # убираем карты с десктопа.
                self.rem_cards_from_desktop()
                # смена раунда
                self.next_round()
                # ходить будет игрок которому передали ход
                self.current_player_id = int(self.player_turn)
            # self.if_human_pause(player_number)
            return
        elif self.result < 0:
            # пропускаем ход (к пассивному игроку)

            ''' Save data about turn experience, with action_idx (self.result) '''
            # if self.pl[self.current_player_id].player_type == 'AI':
            #     self.pl[self.current_player_id].add_turn_experience(self.result)

            self.current_player_id = self.next_player(self.current_player_id)
            # self.show_all_cards(player_number)
            # self.action, self.result = self.pl[player_number].turn()
            return
        pass

    def current_player_passive_action(self) -> None:
        if self.result > 0:
            '''
            выставляем флаг, что _не_ пасуем
            если атакующий игрок пасует и на столе меньше 11 (то есть 10) карт
            '''
            self.pl[self.current_player_id].add_attack_status(self.result)
            self.pl[self.current_player_id].player_cards_onhand_list.remove(self.result)
            self.add_card_2desktop(self.result, self.action, self.current_player_id)

            ''' Save data about turn experience, with action_idx (self.result) '''
            # if self.pl[self.current_player_id].player_type == 'AI':
            self.pl[self.current_player_id].add_turn_experience(self.result)

            # print(f'Подброс от игрока {player_number} - {self.pl[player_number].show_card(result)}')
            self.print_msg(
                f'Подброс от игрока {self.current_player_id} '
                f'{self.pl[self.current_player_id].show_card(self.result)}')
            # print ('Десктоп', self.desktop_list)
            # Пассивный игрок сходил, ставим флаг
            self.set_passive_player_pass_flag(self.current_player_id, False)
            # Переставляем флаг пасующего атакующего на - False
            self.set_attack_player_pass_flag(False)
            # передача отбивающемуся
            self.current_player_id = self.next_player(self.player_turn)
            # self.if_human_pause(player_number)
            return
        elif self.result == 0:
            ''' Save data about turn experience, with action_idx (self.result) '''
            # if self.pl[self.current_player_id].player_type == 'AI':
            self.pl[self.current_player_id].add_turn_experience(self.result)

            # мы пасуем
            self.print_msg(
                f'Игрок {self.current_player_id} {self.pl[self.current_player_id].player_name} тоже пасует')
            # игрок не ходит, ставим флаг
            self.set_passive_player_pass_flag(self.current_player_id, True)
            # print ('Десктоп', self.desktop_list)
            '''
            Пасивный игрок не решает о переходе хода,
            только атакующий. Поскольку мы только подбрасываем
            здесь только передача по кругу следующему игроку
            '''
            self.current_player_id = self.next_player(self.current_player_id)
            # self.if_human_pause(player_number)
            # self.show_all_cards()
            return

        elif self.action == 'Passive' and self.result < 0:
            ''' Save data about turn experience, with action_idx (self.result) '''
            # if self.pl[self.current_player_id].player_type == 'AI':
            #     self.pl[self.current_player_id].add_turn_experience(self.result)

            self.print_msg(
                f'Игрок {self.current_player_id} {self.pl[self.current_player_id].player_name} '
                f'пропускает ход, ждем сигнал от атакующего')
            self.set_passive_player_pass_flag(self.current_player_id, False)
            self.current_player_id = self.next_player(self.current_player_id)
            return
        pass

    def play_episode(self):
        """
        Main game circle
        Show the table. Turning players turns, turning rounds, checking end of the game

        Returns:
            None
        """
        '''
        Проходим по следующему циклу хода
        Атакующий ходит, защищающийся отбивается
        если отбивание произошло, цикл повторяется:
        
        1. пока на столе не будет, 6 пар карт
        2. защищающийся скажет 0 (забираю)
        3. атакующий скажет пас, и пасивный игрок скажет пас тоже
        '''

        self.current_player_id = int(self.player_turn)
        while self.game_circle:
            if (len(self.desktop_list) == 12) or \
                    (self.first_discard and len(self.desktop_list) == 10):

                self.print_msg(
                    f'Игрок {self.next_player(self.player_turn)} '
                    f'{self.pl[self.next_player(self.player_turn)].player_name} '
                    f'отбивается, {"первый сброс" if self.first_discard else ""} ({int(len(self.desktop_list)/2)} пар),'
                    f' карты уходят в сброс {self.pl[self.current_player_id].show_cards_hor(self.desktop_list)}')
                self.add_2graveyard(self.desktop_list)
                # убираем карты с десктопа
                self.rem_cards_from_desktop()
                # Переход хода
                self.next_turn()
                if self.check_end_of_game():
                    continue
                # Переход кона,
                self.next_round()
                # Смена игрока и переход ему хода
                self.current_player_id = int(self.player_turn)
                # self.if_human_pause(player_number)
            self.show_all_cards(self.current_player_id)
            self.action = self.take_action(self.current_player_id)
            self.result = self.pl[self.current_player_id].turn(self.action)

            if self.action == 'Attack':
                self.current_player_attack_action()
                continue
            if self.action == 'Defend':
                self.current_player_defend_action()
                continue
            if self.action == 'Passive':
                self.current_player_passive_action()
                continue
        pass


class Environment(Table):
    def __init__(self,
                 players_qty,
                 games_qty: int,
                 nnmodel=None):
        super().__init__(players_qty)
        self.games_qty: int = games_qty
        self.game_idx: int = 0
        self.game_idxs: list = []
        self.game_winners: list = []
        self.game_losers: list = []
        self.game_times: list = []
        self.game_rounds: list = []
        self.game_turns: int = 0
        self.first_game = True
        self.saved_playing_deck_order = []
        # self.replay_buffer = ExperienceReplay(None)
        self.replay_buffer = ExperienceReplay(30000)
        self.verbose = False
        self.train_process = True
        self.nnmodel = nnmodel
        self.epsilon = 0.0
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.00025)
        # self.loss_funct = tf.keras.losses.Huber()
        self.loss_funct = tf.keras.losses.MSE
        self.compiled_status = False
        if self.nnmodel is not None:
            self.init_nnmodel()
        self.turn_done = False
        pass

    def init_nnmodel(self):
        if not self.compiled_status:
            self.nnmodel.compile(optimizer=self.optimizer,
                                 loss=self.loss_funct)
            self.compiled_status = True
        pass

    def save_deck_order(self) -> None:
        """
        Save current deck shuffle order

        Returns:
            None
        """
        self.saved_playing_deck_order = list(self.hidden_playing_deck_order)
        pass

    def load_deck_order(self) -> None:
        """
        Load saved deck shuffle order

        Returns:
            None
        """
        self.hidden_playing_deck_order = list(self.saved_playing_deck_order)
        pass

    def shuffle(self):
        """
        New shuffle with saving starting shuffle of the deck
        Shuffle the deck of cards
        And save shuffle for using with AI

        Returns:
            None
        """
        # Это лист индексов колоды карт который отражает фактически колоду
        self.hidden_playing_deck_order = list(self.playing_deck.keys())
        # А теперь перемешанную колоду
        random.shuffle(self.hidden_playing_deck_order)
        self.save_deck_order()
        pass

    # Устанавливаем кол-во игроков
    def set_players(self):
        """
        New set players method for training AI
        Setup for types of players

        Returns:
            None
        """
        # Иницианилизируем работу класса игроков и делаем их словарем
        self.print_msg(f'Кол-во игроков: {self.players_number}')
        for player_id in self.players_numbers_lst:
            if player_id == 1:
                # self.pl[1] = Player(1, 1, 0)
                self.pl[1] = Player(1, 2, self.epsilon)
            elif player_id >= 2:
                # Тип 3 - AI
                self.pl[player_id] = AIPlayer(player_id, 3, self.nnmodel, self.epsilon)
            # else:
            #     # Тип 2 - Computer
            #     self.pl[player_id] = Player(player_id, 2)
        pass

    def _reset(self) -> None:
        """
        Reset table (reinit)

        Returns:
            None
        """
        super().__init__(self.players_qty)
        pass

    def __add_report_data(self):
        self.game_idxs.append(self.game_idx)
        self.game_rounds.append(self.game_round)
        self.game_winners.append(self.winner)
        self.game_losers.append(self.looser)
        self.game_times.append(self.time_elapsed)
        pass

    def prepare_new_game(self):
        self.start_time = time.time()
        self.set_table(start_table='new')
        self.first_game = False
        self.save_deck_order()
        pass

    def continue_series(self, start_type='next'):
        if self.verbose:
            self._reset()
            self.verbose = True
        else:
            self._reset()
        self.load_deck_order()
        self.start_time = time.time()
        self.set_table(start_table=start_type)
        if start_type == 'next':
            self.player_turn = self.previous_player(self.game_losers[len(self.game_losers) - 1])
        self.current_player_id = int(self.player_turn)
        pass

    def play_game(self, start_type='next') -> None:
        """
        Play one game

        Args:
            start_type (str):   valid choices for behavior,
                                "new" - 1st run in any series,
                                "same" - using same deck shuffle every time as in 1st run,
                                "next" - using normal behavior for all games
        Returns:
            None
        """
        if self.first_game:
            self.prepare_new_game()
        else:
            self.continue_series(start_type)
        self.game_idx += 1
        msg = f'==========================================\n' \
              f'Игра № {self.game_idx:03d}\n' \
              f'==========================================\n'
        self.print_msg(msg)
        self.play_episode()
        self.__add_report_data()
        pass

    def play_series(self, start_type='next'):
        for game_idx in range(self.games_qty):
            self.play_game(start_type)
            if self.verbose:
                print(f'### player turns')
                for ix, player_id in enumerate(self.episode_players_ranks):
                    print(f'{ix + 1:02d}. {player_id:6d} {self.pl[player_id].game_turn:5d}')

            self.replay_buffer.extend(self.pl[2].episode_buffer)
            if self.debug_verbose > 2:
                buffer = self.replay_buffer.show()
                print('States:')
                for line in buffer[0]:
                    with np.printoptions(precision=3, suppress=True):
                        print(line)
                print('Actions:')
                for line in buffer[1]:
                    with np.printoptions(precision=3, suppress=True):
                        print(line)
                print('Rewards:')
                print(buffer[2])
                print('Dones:')
                print(buffer[3])
                print('Next states:')
                for line in buffer[4]:
                    with np.printoptions(precision=3, suppress=True):
                        print(line)
        print(f'####### rounds win loose   time')
        for ix in range(self.games_qty):
            msg = f'{self.game_idxs[ix]:07d} {self.game_rounds[ix]:6d} {self.game_winners[ix]:3d} ' \
                  f'{self.game_losers[ix]:5d} {self.game_times[ix]:.4f}'
            print(msg)
        print(f'Total playing time: {sum(self.game_times):.4f}')
        print(self.replay_buffer.__len__())
        pass

    def train_episode_AI(self,
                         start_type,
                         nnmodel,
                         epsilon=0.0):

        self.epsilon = epsilon
        self.nnmodel = nnmodel
        if self.first_game:
            self.prepare_new_game()
        else:
            self.continue_series(start_type)
        self.game_idx += 1
        msg = f'==========================================\n' \
              f'Игра № {self.game_idx:03d}\n' \
              f'==========================================\n'
        self.print_msg(msg)
        self.play_episode()
        self.__add_report_data()
        self.game_turns = self.pl[self.episode_players_ranks[0]].game_turn-1
        ''' add data to buffer at training circle if reward is high '''
        # self.replay_buffer.extend(self.pl[2].episode_buffer)
        # last_turn = self.pl[2].episode_buffer[-1]
        # _, _, turn_reward, _, _ = last_turn
        if self.verbose:
            print(f'### player turns')
            for ix, player_id in enumerate(self.episode_players_ranks):
                print(f'{ix + 1:02d}. {player_id:6d} {self.pl[player_id].game_turn:5d} '
                      f'{self.pl[player_id].game_reward}')

        # print(f'####### rounds win loose   time')
        # for ix in range(len(self.game_losers)):
        #     msg = f'{self.game_idxs[ix]:07d} {self.game_rounds[ix]:6d} {self.game_winners[ix]:3d} ' \
        #           f'{self.game_losers[ix]:5d} {self.game_times[ix]:.4f}'
        #     print(msg)

        return self.pl[self.episode_players_ranks[0]].game_reward, self.pl[self.episode_players_ranks[0]].episode_buffer


# Основное тело, перенести потом в инит часть логики
if __name__ == '__main__':
    games_num = 2000
    # while True:
    #     try:
    #         players_num = int(input(f'Введите кол-во игроков (2-6)>'))
    #         if players_num > 6 or players_num < 2:
    #             print("Неправильный ввод")
    #             continue
    #
    #         games_num = int(input(f'Введите кол-во игр в серии (1-9 999 999)>'))
    #         if games_num > 9999999 or games_num < 1:
    #             print("Неправильный ввод")
    #             continue
    #         break
    #     except (TypeError, ValueError):
    #         print("Неправильный ввод")
    players_num = 4
    model = q_model_conv(in_shape=(37, 21 + players_num,), num_actions=37)
    # model = DQNDense(input_shape=(37, 20 + players_num,), output_shape=37)

    fool_game = Environment(players_num,
                            games_num,
                            model)
    fool_game.verbose = True
    fool_game.debug_verbose = True
    # fool_game.play_series(start_type='next')
    count = 0
    total_count = 0
    fool_game.prepare_new_game()
    # while fool_game.game_circle or not fool_game.turn_done:
    # fool_game.play_step(observer_player=2)
    ai_repeat = 'new'
    while count < 1:
        # if ai_repeat == 'new':
        # fool_game.prepare_new_game()
        fool_game.first_game = True
        fool_game._reset()
        # print(ai_repeat)
        fool_game.verbose = True
        reward, episode_buffer = fool_game.train_episode_AI(start_type=ai_repeat, nnmodel=model, epsilon=.99)
        # if reward != 0:
        #     if reward != 1.0:
        #       ai_repeat = 'same'
        #     else:
        #       ai_repeat = 'new'
        # else:
        #     ai_repeat = 'same'
        print(fool_game.episode_players_ranks)
        print(f'Reward: {reward}, Episode buffer length: {len(episode_buffer)}')
        # if reward != 0:
        count += 1
        #     print(f'Counted: {count}/{total_count}')
        # total_count += 1

    # fool_game.train_episode_AI(start_type='next', epsilon=.5)
