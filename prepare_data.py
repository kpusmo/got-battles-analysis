import csv
from sklearn.model_selection import train_test_split

file = 'in/battles.csv'
y_key = 'attacker_outcome'
classes = {
    'loss': 0,
    'win': 1,
}
labels = list(classes.keys())
battle_types = {
    'pitched battle': 0,
    'ambush': 1,
    'siege': 2,
    'razing': 3
}
regions = {
    'Beyond the Wall': 0,
    'The North': 1,
    'The Iron Islands': 2,
    'The Riverlands': 3,
    'The Vale of Arryn': 4,
    'The Westerlands': 5,
    'The Crownlands': 6,
    'The Reach': 7,
    'The Stormlands': 8,
    'Dorne': 9
}
houses = {
    'Lannister': 0,
    'Stark': 1,
    'Greyjoy': 2,
    'Bolton': 3,
    'Baratheon': 4,
    'Darry': 5,
    'Brotherhood without Banners': 6,
    'Frey': 7,
    'Free folk': 8,
    'Brave Companions': 9,
    'Balon/Euron Greyjoy': 10,
    'Bracken': 11,
    'Tully': 12,
    'Blackwood': 13,
    'Tyrell': 14,
    'Night\'s Watch': 15,
    'Mallister': 16,
    'folk': 17
}
feature_names = ['attacker_1', 'defender_1', 'battle_type', 'summer', 'region']


def choose_keys(d, keys):
    return {x: d[x] for x in d if x in keys}


def load_data_from_csv(file):
    # todo use pandas for this!!!
    with open(file) as battles_file:
        battles_data = csv.DictReader(battles_file, quotechar='"', delimiter=',')
        x = []
        y = []
        for row in battles_data:
            y.append(classes[row[y_key]])
            row = choose_keys(row, feature_names)
            row['battle_type'] = battle_types[row['battle_type']]
            row['region'] = regions[row['region']]
            row['attacker_1'] = houses[row['attacker_1']]
            row['defender_1'] = houses[row['defender_1']]
            row['summer'] = int(row['summer'])
            x.append(list(row.values()))
        return x, y


def print_dict(dictionary):
    keys = list(dictionary.keys())
    values = list(dictionary.values())
    for i in range(0, len(keys)):
        print('\t' + keys[i] + ': ' + str(values[i]))


def print_data_maps():
    print("classes:")
    print_dict(classes)
    print("battle_types:")
    print_dict(battle_types)
    print("regions:")
    print_dict(regions)
    print("houses:")
    print_dict(houses)
    print("summer: 1/0 (yes/no)")
    print("train data length: {}\ntest data length: {}\n".format(len(x_train), len(x_test)))


# Split dataset into training and test set
x_data, y_data = load_data_from_csv(file)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, shuffle=False)
