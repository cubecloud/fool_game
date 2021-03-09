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

from random import choice, shuffle
import copy
import time

__version__ = 0.0013


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
        [1] - ранг [6..14] (включительно)
        Статус:
        [2] принадлежность игроку [1..6] (включительно)
            - 0 - лежит в колоде
            - 1 - карта находится у игрока с номером 1
            - 2 - карта находится у игрока с номером 2
            - 3 - карта находится у игрока с номером 3
            - 4 - карта находится у игрока с номером 4
            - 5 - карта находится у игрока с номером 5
            - 6 - карта находится у игрока с номером 6
        [3] Статус карты
            - 0 - лежит в колоде
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
    player_deck = {
        # Это инициализация масти пики - 9 карт от 6 пик до туза пик
        1: [1, 1, 0, 0, 0, 0, 6], 2: [1, 2, 0, 0, 0, 0, 7], 3: [1, 3, 0, 0, 0, 0, 8],
        4: [1, 4, 0, 0, 0, 0, 9], 5: [1, 5, 0, 0, 0, 0, 10], 6: [1, 6, 0, 0, 0, 0, 11],
        7: [1, 7, 0, 0, 0, 0, 12], 8: [1, 8, 0, 0, 0, 0, 13], 9: [1, 9, 0, 0, 0, 0, 14],
        # Это инициализация масти крести - 9 карт от 6 крестей до туза крестей
        10: [2, 1, 0, 0, 0, 0, 6], 11: [2, 2, 0, 0, 0, 0, 7], 12: [2, 3, 0, 0, 0, 0, 8],
        13: [2, 4, 0, 0, 0, 0, 9], 14: [2, 5, 0, 0, 0, 0, 10], 15: [2, 6, 0, 0, 0, 0, 11],
        16: [2, 7, 0, 0, 0, 0, 12], 17: [2, 8, 0, 0, 0, 0, 13], 18: [2, 9, 0, 0, 0, 0, 14],
        # Это инициализация масти буби - 9 карт от 6 бубей до туза бубей
        19: [3, 1, 0, 0, 0, 0, 6], 20: [3, 2, 0, 0, 0, 0, 7], 21: [3, 3, 0, 0, 0, 0, 8],
        22: [3, 4, 0, 0, 0, 0, 9], 23: [3, 5, 0, 0, 0, 0, 10], 24: [3, 6, 0, 0, 0, 0, 11],
        27: [3, 7, 0, 0, 0, 0, 12], 26: [3, 8, 0, 0, 0, 0, 13], 25: [3, 9, 0, 0, 0, 0, 14],
        # Это инициализация масти черви - 9 карт от 6 червей до туза червей
        28: [4, 1, 0, 0, 0, 0, 6], 29: [4, 2, 0, 0, 0, 0, 7], 30: [4, 3, 0, 0, 0, 0, 8],
        31: [4, 4, 0, 0, 0, 0, 9], 32: [4, 5, 0, 0, 0, 0, 10], 33: [4, 6, 0, 0, 0, 0, 11],
        34: [4, 7, 0, 0, 0, 0, 12], 35: [4, 8, 0, 0, 0, 0, 13], 36: [4, 9, 0, 0, 0, 0, 14],
    }

    suit_range = {'П': (1, 10), 'К': (10, 19), 'Б': (19, 28), 'Ч': (28, 37)}
    rank_names = {1: '6', 2: '7', 3: '8', 4: '9', 5: '10', 6: 'В', 7: 'Д', 8: 'К', 9: 'Т'}
    suit_chars = {1: 'П', 2: 'К', 3: 'Б', 4: 'Ч'}
    suits_names = {1: "Пики", 2: "Крести", 3: "Бубны", 4: "Черви"}
    suits_icons = {'П': '\u2660', 'К': '\u2663', 'Б': '\u2666', 'Ч': '\u2665'}

    def change_card_status(self, index, status):
        self.player_deck[index][2:6] = status

    def get_cardinfo(self, index):
        return self.player_deck[index]

    def get_current_status(self, index):
        return self.player_deck[index][2:6]

    def change_card_weight(self, index, new_weight):
        self.player_deck[index][6] = new_weight

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

    def show_cards_hor(self, c_list):
        output = str()
        for card in c_list:
            output += (str(self.show_card(card)) + ' ')
            # print(output)
        return output

    def show_cards_vert_numbered(self, c_list):
        cards_on_hand = 1
        for card in c_list:
            print(f'{cards_on_hand}. ' + self.show_card(card))
            cards_on_hand += 1

    pass


class Player(Deck):
    def __init__(self, player_number, player_type):

        self.player_number = player_number
        self.player_types = {1: 'Human', 2: 'Computer'}
        self.player_type = self.player_types[player_type]
        # если человек то запрашиваем имя
        if self.player_type == self.player_types[1]:
            # self.player_name = self.player_types[player_type]
            self.ask_for_name()
        else:
            self.player_name = 'Computer №' + str(self.player_number)
        self.player_cards_onhand_list = list()
        self.game_round = 0
        self.player_turn = 0
        self.players_number = 0
        self.desktop_list = list()
        self.passive_player_pass_flag = False
        self.attack_player_pass_flag = False
        self.action = None
        self.trump_index = None
        self.trump_char: str = ''
        self.trump_range = tuple
        pass

    def change_game_round(self, game_round):
        self.game_round = game_round
        pass

    def change_player_turn(self, turn):
        self.player_turn = turn
        pass

    def add_graveyard_status(self, index):
        status = self.get_current_status(index)
        # graveyard status
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

    # Возвращает лист возможных ответов (карт) на защиту
    def get_validated_defend_list(self, index):
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

    # возвращает кол-во карт необходимое взять в руку игроку из колоды.
    def check_hand_before_round(self):
        temp = 6 - (len(self.player_cards_onhand_list))
        if temp < 0:
            temp = 0
        return temp

    #  это ход игрока
    def turn(self, action):
        self.action = action
        if self.player_type == 'Computer':
            result = self.analyze()
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
            result = -1
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

    def attacking(self):
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

    def defending(self):
        if not self.attack_player_pass_flag or not self.passive_player_pass_flag:
            check_parity = (len(self.desktop_list) + 1) % 2
            if check_parity == 0:
                # Последняя карта в десктоп листе
                defending_card = self.desktop_list[(len(self.desktop_list)) - 1]
                # print ('Defending', defending_card)
                defend_list = self.get_validated_defend_list(defending_card)
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

    # Атака пассивного компьютера/игрока
    def passive_attacking(self):
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

    # Возвращает индекс карты или в случае
    def analyze(self):
        if self.action == 'Attack':
            r_index = self.attacking()
        elif self.action == 'Defend':
            r_index = self.defending()
        elif self.action == 'Passive':
            r_index = self.passive_attacking()
        return r_index

    def set_trump(self, index):
        self.trump_index = index
        self.trump_char = self.what_suit(index)
        status = self.get_current_status(index)
        # trump in current game status
        status[1] = 4
        self.change_card_status(index, status)
        self.add_weight_2suit(self.trump_char, 100)
        self.trump_range = range(self.suit_range[self.trump_char][0], self.suit_range[self.trump_char][1])
        pass

    def get_deck(self, indeck):
        self.player_deck = copy.deepcopy(indeck)
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
        print('Козырь: ' + self.show_card(self.trump_index))
        pass


class Table:
    def __init__(self, players_number):
        # кол-во игроков
        self.winner = 0
        self.looser = 0
        self.dd = Deck()
        self.players_number = players_number
        self.playing_deck = self.dd.player_deck
        self.desktop_list: list = []
        self.end_of_deck = False
        self.result = 0
        self.players_numbers_lst: list = []
        self.hidden_deck_index: int = -1
        self.hidden_playing_deck_order: list = []
        self.trump_index: int = -1
        self.player_turn: int = 0
        self.game_round: int = 0
        self.action: str = str()
        self.attack_player_empty_hand_flag = False
        self.pl: dict = {}
        for i in range(1, self.players_number + 1):
            self.players_numbers_lst.append(i)

    # передаем индекс карты из списка self.hidden_playing_deck_order,
    # ссылаясь на индекс верхней карты колоды
    def current_card_index(self):
        return self.hidden_playing_deck_order[self.hidden_deck_index]

    # Пометить каждую карту из переданного списка,
    # в колоде каждого игрока как находящуюся в сбросе
    def add_2graveyard(self, g_list):
        for index in g_list:
            for player_number in self.players_numbers_lst:
                # Убрать карту со стола (поменять статус стола и принадлежности на 'Сброс')
                # Добавим карту в свою базу знаний
                self.pl[player_number].add_graveyard_status(index)
        pass

    def add_card_2desktop(self, index):
        self.desktop_list.append(index)
        for player_number in self.players_numbers_lst:
            self.pl[player_number].desktop_list = self.desktop_list
        pass

    def rem_cards_from_desktop(self):
        self.desktop_list.clear()
        for player_number in self.players_numbers_lst:
            self.pl[player_number].desktop_list = self.desktop_list
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
            print(self.pl[1].show_cards_hor(desktop_list_1))
        else:
            print()
        if len(desktop_list_2) > 0:
            print(self.pl[1].show_cards_hor(desktop_list_2))
        else:
            print()
        pass

    # добавить карту в руку игрока (раздача, забрать
    # если не отбился, или из колоды после отбоя)
    def add_card_2player_hand(self, p_number):
        # Добавим карту в руку игрока
        if self.hidden_deck_index <= 35 and not self.end_of_deck:
            self.pl[p_number].get_card(self.current_card_index())
            # Индекс карты в дек листе меняем на следующую карту
            self.hidden_deck_index += 1
            if self.hidden_deck_index == 36:
                self.hidden_deck_index = 35
                self.end_of_deck = True
        else:
            self.end_of_deck = True
        pass

    def add_cardlist_2player_hand(self, p_number, c_list):
        for index in c_list:
            # взять карту из листа в руку
            self.pl[p_number].get_card(index)
            for player_number in self.players_numbers_lst:
                if player_number != p_number:
                    status = self.pl[player_number].get_current_status(index)
                    # player number
                    status[0] = p_number
                    # on hand status
                    status[1] = 3
                    self.pl[player_number].change_card_status(index, status)
                else:
                    self.pl[player_number].add_player_status(index)
        pass

    # Установить козыря для всех игроков
    # (фактически показать его всем)
    def add_trump_card(self):
        for player_number in self.players_numbers_lst:
            # Поменять статус на 'Козырь')
            # Добавим карту в базу знаний всех игроков
            # Сделать с индексом козыря (в деке по этому номеру лежит последняя козырная карта в колоде)
            self.pl[player_number].set_trump(self.current_card_index())
        self.trump_index = self.current_card_index()
        # перенести в скрытом листе, открытого козыря последним в список, перемешанной деки
        # чтобы он был последним
        self.hidden_playing_deck_order.remove(self.trump_index)
        self.hidden_playing_deck_order.append(self.trump_index)
        # Индекс карты в дек листе меняем на следующую карту
        if self.hidden_deck_index < 35:
            self.hidden_deck_index += 1
        pass

    # Выбор игрока с первым ходом
    # логика: сначалас ищем игрока с наименьшим козырем,
    # если такого нет ищется игрок с наименьшей картой (по рангу масти)
    def first_turn_choice(self):
        # print ("Идет выбор ходящего первым")
        min_card_index: dict = {}
        for player_number in self.players_numbers_lst:
            # print('Игрок',player_number, self.show_cards_list(self.pl[player_number].trumps_from_hand()))
            min_card_index[player_number] = self.pl[player_number].lowest_trump_from_hand()
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
            for player_number in self.players_numbers_lst:
                # если лист пуст ищем дальше по циклу
                if not min_card_index[player_number]:
                    continue
                # если список не пустой проверяем значение и записываем если больше
                elif min_card_index[player_number] > check_card:
                    check_card = min_card_index[player_number]
                    check_player = player_number
            # если есть игрок с козырем то выводим
            if check_card != 0:
                min_card_player = check_player
            # Игрока с козырем нет - ищем _любую_ самую младшую карту у игроков (пики, крести, бубны, червы)
            else:
                for playerNumber1 in self.players_numbers_lst:
                    # print('Игрок',playerNumber1,
                    # self.show_cards_list(self.pl[playerNumber1].player_cards_onhand_list))
                    min_card_index[playerNumber1] = self.pl[playerNumber1].lowest_from_hand()
                min_card_player = (min(min_card_index.items(), key=lambda x: x[1])[0])
        # print(min_card_index)
        # print (f'Минимальная карта у игрока {min_card_player},
        # это карта', self.show_card(min_card_index[min_card_player]))
        return min_card_player, min_card_index[min_card_player]

    # Устанавливаем кол-во игроков
    def set_players(self):
        # Иницианилизируем работу класса игроков и делаем их словарем
        print(f'Кол-во игроков: {self.players_number}')
        for i in self.players_numbers_lst:
            if i == 1:
                # self.pl[1] = Player(1, 1)
                self.pl[1] = Player(1, 2)
            else:
                # Тип 2 - Компьютер
                self.pl[i] = Player(i, 2)
        pass

    # Инициализируем словарь (массив) с деками игроков
    def table_init(self):
        for player_number in self.players_numbers_lst:
            # заносим в словари деки игроков
            self.pl[player_number].get_deck(self.playing_deck)
            self.pl[player_number].players_number = self.players_number
        # устанавливаем индекс карты из колоды на 1
        # это индекс для ###### self.hidden_playing_deck_order ######
        self.hidden_deck_index = 0
        pass

    # мешаем колоду
    def shuffle(self):
        # Это лист индексов колоды карт который отражает фактически колоду
        self.hidden_playing_deck_order = list(self.playing_deck.keys())
        # А теперь перемешанную колоду
        shuffle(self.hidden_playing_deck_order)

    def show_first_turn_card(self):
        player, index = self.first_turn_choice()
        self.player_turn = player
        print(f'Ходит игрок №{player} {self.pl[player].player_name}, у него меньшая карта',
              self.pl[player].show_card(index))
        status = self.pl[player].get_current_status(index)
        # player number
        status[0] = player
        # on hand status
        status[1] = 3
        for player_number in self.players_numbers_lst:
            self.pl[player_number].change_card_status(index, status)
            # self.pl[player_number].change_card_status(index, 'P' + str(player))
        pass

    def set_table(self):
        self.set_players()
        self.shuffle()
        self.table_init()
        # print(self.hidden_playing_deck_order)
        # раздача
        for i in range(6):
            for player_number in self.players_numbers_lst:
                # добавляем 1 карту каждому игроку
                self.add_card_2player_hand(player_number)
        self.add_trump_card()
        self.show_first_turn_card()
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

    # проверяем есть ли выигрывший и проигравший
    def check_end_of_game(self):
        if self.is_this_end_of_game():
            self.congratulations()
            exit(0)
        pass

    def is_this_end_of_game(self):
        # Проверка окончания игры если какой_либо игрок закончил игру.
        result = False
        for player_number in self.players_numbers_lst:
            if self.if_player_hand_and_deck_empty(player_number):
                print(
                    f'Игрок № {player_number}, заканчивает игру. Остается {len(self.players_numbers_lst) - 1} игроков')
                self.show_desktop()
                self.rem_cards_from_desktop()
                if self.players_number == 2:
                    self.looser = self.next_player(player_number)
                    result = True
                self.one_more_is_out(player_number)
        return result

    def congratulations(self):
        msg = f'Победитель игрок №{self.winner}\nПроигравший игрок №{self.looser}\n' \
              f'Игра закончена за {self.game_round} раундов и {time.time() - start_time:.2f} сек'
        print(msg)
        pass

    # показать все карты на столе
    def show_all_cards(self, player_number):
        self.if_human_pause(player_number)
        # Для теста
        # for i in self.players_numbers_lst:
        #     print(f'Игрок {i}', self.pl[i].show_cards_hor(self.pl[i].player_cards_onhand_list))
        print(f'Раунд № {self.game_round}')
        print('В колоде карт: ' + str(35 - self.hidden_deck_index))
        print(f'Заход игрока {self.player_turn}, {self.pl[self.player_turn].player_name}')
        print(
            f'Отбой игрока {self.next_player(self.player_turn)}, '
            f'{self.pl[self.next_player(self.player_turn)].player_name}')
        self.pl[1].show_trump()
        # self.pl[1].show_cards_vert_numbered(self.pl[1].player_cards_onhand_list)
        if len(self.desktop_list) > 0:
            print('На столе')
            self.show_desktop()
        else:
            print('Стол пуст')
            print()
        pass

    def next_turn(self):
        self.player_turn = self.next_player(self.player_turn)
        for player_number in self.players_numbers_lst:
            self.pl[player_number].change_player_turn(self.player_turn)
        pass

    def next_player(self, p_number):
        # Индекс должен быть меньше чем в реальный player_number
        index = self.players_numbers_lst.index(p_number)
        if index + 1 > len(self.players_numbers_lst) - 1:
            index = 0
            return self.players_numbers_lst[index]
        else:
            index += 1
            return self.players_numbers_lst[index]
        pass

    def previous_player(self, p_number):
        index = self.players_numbers_lst.index(p_number)
        if index - 1 == -1:
            index = len(self.players_numbers_lst) - 1
            return self.players_numbers_lst[index]
        else:
            index -= 1
            return self.players_numbers_lst[index]
        pass

    '''
    проверяем есть ли флаг {пасс} у атакующего игрока
    '''

    def check_attack_player_pass_flag(self):
        if self.pl[self.player_turn].attack_player_pass_flag:
            return True
        else:
            return False
        pass

    def set_attack_player_pass_flag(self, flag):
        """
        Внимание! Устанавливает флаг, что
        атакующий игрок пасует - ВСЕМ игрокам
        """
        for player_number in self.players_numbers_lst:
            self.pl[player_number].attack_player_pass_flag = flag
        pass

    def set_passive_player_pass_flag(self, player_number, flag):
        """
        Внимание! устанавливает флаг только
        текущему игроку (у кого сейчас ход)
        """
        self.pl[player_number].passive_player_pass_flag = flag

    def check_passive_player_pass_flag(self):
        """
        проверяем есть ли флаг пасс у пассивных игроков
        проверка идет у атакующего игрока
        """
        # если игроков 2 то флаг True
        if self.players_number == 2:
            return True
        # если игроков 3 то проверяем флаг у 1 игрока
        # поскольку проверка идет у атакующего игрока,
        # 3-й игрок, это предыдущий атакующему
        elif self.players_number == 3:
            if self.pl[self.previous_player(self.player_turn)].passive_player_pass_flag:
                return True
            else:
                return False
        # если игроков 4 то проверяем флаг у 2 игроков
        # поскольку проверка идет у атакующего игрока,
        # 3-й и 4-й игрок, это предыдущий атакующему
        # и передыдущий, предыдущему атакующему
        elif self.players_number == 4:
            if self.pl[self.previous_player(self.player_turn)].passive_player_pass_flag \
                    and self.pl[self.previous_player(self.previous_player(self.player_turn))].passive_player_pass_flag:
                return True
            else:
                return False

    def next_round(self):
        self.game_round += 1
        self.set_attack_player_pass_flag(False)
        for player_number in self.players_numbers_lst:
            self.set_passive_player_pass_flag(player_number, False)
            if not self.end_of_deck:
                for i in range(self.pl[player_number].check_hand_before_round()):
                    self.add_card_2player_hand(player_number)
            self.pl[player_number].change_game_round(self.game_round)
            self.pl[player_number].change_player_turn(self.player_turn)

    def if_human_pause(self, player_number):
        # flag = False
        # if self.end_of_deck:
        #     for player_number in self.players_numbers_lst:
        #         if len(self.pl[player_number].player_cards_onhand_list) < 2:
        #             flag = True
        # if flag:
        #     time.sleep(2)
        #     self.clear_screen()

        if self.pl[player_number].player_type == 'Computer':
            # time.sleep(3)
            self.clear_screen()
        else:
            # time.sleep(2)
            self.clear_screen()

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

    # Показать стол (передача хода здесь)
    def show_table(self):
        #   Устанавливаем ход
        self.game_round = 0
        self.next_round()
        # print(f'Раунд № {self.game_round}')
        # self.pl[1].show_cards_vert_numbered(self.pl[1].player_cards_onhand_list)
        # print('0. Пас/забрать')
        # self.show_trump()
        # Проходим по следующему циклу хода
        # Атакующий ходит, защищающийся отбивается
        # если отбивание произошло, цикл повторяется,
        # 1. пока на столе не будет, 6 пар карт
        # 2. защищающийся скажет 0 (забираю)
        # 3. атакующий скажет пас, и пасивный игрок скажет пас тоже
        circle = True
        player_number = self.player_turn
        while circle:
            if len(self.desktop_list) == 12:
                print(
                    f'Игрок {self.next_player(self.player_turn)} '
                    f'{self.pl[self.next_player(self.player_turn)].player_name} '
                    f'отбивается (6 пар), карты уходят в сброс',
                    self.pl[player_number].show_cards_hor(self.desktop_list))
                self.add_2graveyard(self.desktop_list)
                # убираем карты с десктопа
                self.rem_cards_from_desktop()
                # Переход хода
                self.next_turn()
                self.check_end_of_game()
                # Переход кона,
                self.next_round()
                # Смена игрока и переход ему хода
                player_number = self.player_turn
                # self.if_human_pause(player_number)
            else:
                self.show_all_cards(player_number)
                self.action = self.take_action(player_number)
                self.result = self.pl[player_number].turn(self.action)
                if self.action == 'Attack' and self.result > 0:
                    self.attack_player_empty_hand_flag = False
                    self.pl[player_number].add_attack_status(self.result)
                    self.pl[player_number].player_cards_onhand_list.remove(self.result)
                    self.add_card_2desktop(self.result)
                    # print(f'Ход игрока {player_number}
                    # {self.pl[player_number].player_name} - {self.pl[player_number].show_card(result)}')
                    print(
                        f'Ход игрока {player_number} {self.pl[player_number].player_name} под игрока '
                        f'{self.next_player(player_number)} {self.pl[player_number].show_card(self.result)}')
                    # print (f'Атака игрока {player_number}',self.pl[player_number].show_card(result))
                    # print ('Десктоп', self.desktop_list)
                    # передача по кругу следующему игроку
                    player_number = self.next_player(player_number)
                    # выставляем флаг, что мы не пасуем
                    self.set_attack_player_pass_flag(False)
                    # print ('PN',player_number, 'PT',self.player_turn)
                    # self.show_all_cards()
                    # self.if_human_pause(player_number)
                    # if self.if_player_hand_and_deck_empty(player_number):
                    #     self.attack_player_empty_hand_flag = True
                    continue
                elif self.action == 'Attack' and self.result == 0:
                    # если всего 2 игрока
                    if self.players_number == 2:
                        # Карты уходят в сброс того что входит
                        print(
                            f'Игрок {player_number} {self.pl[player_number].player_name} пасует, карты уходят в сброс',
                            self.pl[player_number].show_cards_hor(self.desktop_list))
                        self.add_2graveyard(self.desktop_list)
                        # убираем карты с десктопа
                        self.rem_cards_from_desktop()
                        # Переход хода
                        self.next_turn()
                        self.check_end_of_game()
                        # Переход кона,
                        self.next_round()
                        player_number = self.player_turn
                        continue
                    # Проверяем флаги, что мы уже пасовали
                    # и пассивные игроки пасовали и можно отправить карты в сброс
                    elif self.players_number > 2 and \
                            self.check_passive_player_pass_flag() and \
                            self.check_attack_player_pass_flag():
                        # Карты уходят в сброс
                        print(f'Карты уходят в сброс', self.pl[player_number].show_cards_hor(self.desktop_list))
                        self.add_2graveyard(self.desktop_list)
                        # убираем карты с десктопа
                        self.rem_cards_from_desktop()
                        # Переход хода
                        self.next_turn()
                        # если этот игрок
                        self.check_end_of_game()
                        # Переход кона,
                        self.next_round()
                        # Смена игрока и переход ему хода
                        player_number = self.player_turn
                        continue
                    # если в сброс карты не отправились, то....
                    elif self.players_number > 2:
                        # мы пасуем, но может сходить следующий игрок.
                        # Поэтому мы передаем ход через 1 игрока (отбивающегося)
                        print(f'Игрок {player_number} {self.pl[player_number].player_name} пасует, можно подбрасывать')
                        player_number = self.next_player(self.next_player(player_number))
                        # и выставляем флаг, что мы пасуем
                        self.set_attack_player_pass_flag(True)
                    continue
                # self.show_all_cards(player_number)
                # self.action, self.result = self.pl[player_number].turn()
                if self.action == 'Defend' and self.result > 0:
                    self.pl[player_number].add_defending_status(self.result)
                    # print(self.pl[player_number].player_cards_onhand_list, result)
                    self.pl[player_number].player_cards_onhand_list.remove(self.result)
                    self.add_card_2desktop(self.result)
                    # print(
                    #     f'Игрок {player_number}
                    #     {self.pl[player_number].player_name} отбивается - {self.pl[player_number].show_card(result)}')
                    print(
                        f'Игрок {player_number} {self.pl[player_number].player_name} '
                        f'отбивается {self.pl[player_number].show_card(self.result)}')
                    # print ('Десктоп', self.desktop_list)

                    # print ('PN',player_number, 'PT',self.player_turn)
                    # self.if_human_pause(player_number)
                    # если в руке больше нет карт на отбой и в колоде пусто ИЛИ
                    # если нам не достанется ничего из колоды при раздаче и в руке нет больше карт на отбой
                    # - мы выходим из игры и нас исключают из списка играющих. Остальные играют дальше
                    # Переход хода идет на следующего игрока (следующего за отбивающимся)
                    if self.if_player_hand_and_deck_empty(player_number) or \
                            ((35 - self.hidden_deck_index) < len(self.desktop_list) // 2 and
                             (len(self.pl[player_number].player_cards_onhand_list) == 0)):
                        if self.players_number != 2:
                            self.next_turn()
                        self.next_turn()
                        self.check_end_of_game()
                        self.next_round()
                        player_number = self.player_turn
                        continue
                    # передача по кругу следующему игроку
                    player_number = self.next_player(player_number)
                    continue
                elif self.action == 'Defend' and self.result == 0:
                    # Если забираем карты (нет отбоя)
                    print(f'Игрок {player_number} {self.pl[player_number].player_name} забирает',
                          self.pl[player_number].show_cards_hor(self.desktop_list))
                    self.add_cardlist_2player_hand(player_number, self.desktop_list)
                    # проверяем на наличие карт
                    # если игроков 2 и пас то, если 2 игрока просто следующий кон,
                    # но игрок остается тот-же
                    if self.players_number == 2:
                        self.check_end_of_game()
                        # убираем карты с десктопа
                        self.rem_cards_from_desktop()
                        # Переход кона,
                        self.next_round()
                        # Перехода хода нет - ход остается у прежнего игрока
                        # передаем ход ему обратно
                        player_number = self.player_turn
                    elif self.players_number > 2:
                        # если игроков больше 2-х,
                        # смена раунда
                        self.next_turn()
                        self.next_turn()
                        self.check_end_of_game()
                        # убираем карты с десктопа
                        self.rem_cards_from_desktop()
                        self.next_round()
                        # переход хода на 2 вперед. Мы забрали карты со стола
                        # ходить будет игрок которому передали ход
                        player_number = self.player_turn
                    # self.if_human_pause(player_number)
                    elif self.action == 'Defend' and self.result < 0:
                        # пропускаем ход (к пассивному игроку)
                        player_number = self.next_player(player_number)
                    # self.show_all_cards(player_number)
                    # self.action, self.result = self.pl[player_number].turn()
                    continue
                if self.action == 'Passive' and self.result > 0:
                    # выставляем флаг, что _не_ пасуем
                    # если атакующий игрок пасует и на столе меньше 11 (то есть 10) карт
                    self.pl[player_number].add_attack_status(self.result)
                    self.pl[player_number].player_cards_onhand_list.remove(self.result)
                    self.add_card_2desktop(self.result)
                    # print(f'Подброс от игрока {player_number} - {self.pl[player_number].show_card(result)}')
                    print(f'Подброс от игрока {player_number} {self.pl[player_number].show_card(self.result)}')
                    # print ('Десктоп', self.desktop_list)
                    # Пассивный игрок сходил, ставим флаг
                    self.set_passive_player_pass_flag(player_number, False)
                    # Переставляем флаг пасующего атакующего на - False
                    self.set_attack_player_pass_flag(False)
                    # передача отбивающемуся
                    player_number = self.next_player(self.player_turn)
                    # self.if_human_pause(player_number)
                    continue
                elif self.action == 'Passive' and self.result == 0:
                    # мы пасуем
                    print(f'Игрок {player_number} {self.pl[player_number].player_name} тоже пасует')
                    # игрок не ходил, ставим флаг
                    self.set_passive_player_pass_flag(player_number, True)
                    # print ('Десктоп', self.desktop_list)
                    # Пасивный игрок не решает о переходе хода,
                    # только атакующий. Поскольку мы только подбрасываем
                    # здесь только передача по кругу следующему игроку
                    player_number = self.next_player(player_number)
                    # self.if_human_pause(player_number)
                    # self.show_all_cards()
                    continue
                elif self.action == 'Passive' and self.result < 0:
                    print(
                        f'Игрок {player_number} {self.pl[player_number].player_name} '
                        f'пропускает ход, ждем сигнал от атакующего')
                    self.set_passive_player_pass_flag(player_number, False)
                    player_number = self.next_player(player_number)
                    continue


# Основное тело, перенести потом в инит часть логики
if __name__ == '__main__':
    while True:
        try:
            players_number = int(input(f'Введите кол-во игроков 2-6>'))
            if players_number > 6 or players_number < 2:
                print("Неправильный ввод")
                continue
            break
        except (TypeError, ValueError):
            print("Неправильный ввод")
    fool_game = Table(players_number)
    start_time = time.time()
    fool_game.set_table()
    fool_game.show_table()
