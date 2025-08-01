# pip install ruamel.yaml
import csv
import yaml
from collections import defaultdict
from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import DoubleQuotedScalarString as dq

def csv_to_yaml(csv_path, output_yaml_path):
    # 从CSV读取比赛数据
    matches = defaultdict(lambda: {'maps': [], 'teams': None})

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            match_id = row['MatchId']
            map_idx = int(row['MapIdx'])

            # 只处理每个地图的第一行记录（避免重复）
            if map_idx > len(matches[match_id]['maps']):
                scores = list(map(int, row['Scores'].split('-')))
                team1, team2 = row['Team'], row['vs']

                # 存储队伍顺序（仅在第一张地图时设置）
                if matches[match_id]['teams'] is None:
                    matches[match_id]['teams'] = (team1, team2)

                # 添加地图数据
                matches[match_id]['maps'].append({
                    'name': row['Map'],
                    'score': scores
                })

    # 构建YAML数据结构
    yaml_data = {
        'regular_season': [],
        'playoffs': []
    }

    # 按MatchId排序处理比赛
    for match_id in sorted(matches.keys(), key=int):
        data = matches[match_id]
        team1, team2 = data['teams']

        # 计算总比分
        total_score = [0, 0]
        for map_data in data['maps']:
            if map_data['score'][0] > map_data['score'][1]:
                total_score[0] += 1
            else:
                total_score[1] += 1

        # 添加比赛数据
        yaml_data['regular_season'].append({
            'match': [team1, team2, total_score],
            'maps': data['maps']
        })

    # 写入YAML文件
    with open(output_yaml_path, 'w', encoding='utf-8') as f:
        f.write("# https://www.vlr.gg/event/matches/2499/vct-2025-china-stage-2/\n")
        yaml.dump(yaml_data, f, sort_keys=False,
                  allow_unicode=True, default_flow_style=None)


def load_team_mapping(team_name_path):
    """加载队伍名称映射（全名到缩写）"""
    with open(team_name_path, 'r', encoding='utf-8') as f:
        team_data = yaml.safe_load(f)
    # 创建全名到缩写的反向映射
    full_to_abbr = {v: k for k, v in team_data.items()}
    return full_to_abbr


def load_match_list(match_list_path, full_to_abbr):
    """加载比赛列表并转换为缩写"""
    with open(match_list_path, 'r', encoding='utf-8') as f:
        match_data = yaml.safe_load(f)

    # 按index排序并转换为缩写
    sorted_matches = []
    for match_id, details in match_data.items():
        abbr_teams = [full_to_abbr.get(team, team)
                      for team in details['teams']]
        sorted_matches.append((
            details['index'],
            match_id,
            abbr_teams,
            details['status'],
            details['scores']
        ))

    # 按index排序
    sorted_matches.sort(key=lambda x: x[0])
    return sorted_matches


def csv_to_yaml_with_coming(csv_path, match_list_path, team_name_path, output_yaml_path):
    # 加载队伍名称映射
    full_to_abbr = load_team_mapping(team_name_path)

    # 加载并处理比赛列表
    all_matches = load_match_list(match_list_path, full_to_abbr)

    # 从CSV读取已完成的比赛数据
    completed_matches = defaultdict(lambda: {'maps': []})
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            match_id = row['MatchId']
            map_idx = int(row['MapIdx'])

            # 只处理每个地图的第一行记录
            if map_idx > len(completed_matches[match_id]['maps']):
                scores = list(map(int, row['Scores'].split('-')))
                completed_matches[match_id]['maps'].append({
                    'name': row['Map'],
                    'score': scores
                })

    # 构建YAML数据结构
    yaml_data = {'regular_season': [], 'playoffs': []}

    # 按顺序处理所有比赛
    for idx, match_id, teams, status, scores in all_matches:
        if status == 'Completed':
            # 计算总比分
            total_score = [0, 0]
            for map_data in completed_matches[match_id]['maps']:
                if map_data['score'][0] > map_data['score'][1]:
                    total_score[0] += 1
                else:
                    total_score[1] += 1

            yaml_data['regular_season'].append({
                'match': teams + [total_score],
                'maps': completed_matches[match_id]['maps']
            })
        else:  # 未开始的比赛
            yaml_data['regular_season'].append({
                'match': teams + [None],
                'maps': None
            })

    # 写入YAML文件
    with open(output_yaml_path, 'w', encoding='utf-8') as f:
        f.write("# https://www.vlr.gg/event/matches/2499/vct-2025-china-stage-2/\n")
        yaml.dump(yaml_data, f, sort_keys=False,
                  allow_unicode=True, default_flow_style=None)

def csv_to_yaml_with_coming_ryaml(csv_path, match_list_path, team_name_path, output_yaml_path):
    # 加载队伍名称映射
    full_to_abbr = load_team_mapping(team_name_path)

    # 加载并处理比赛列表
    all_matches = load_match_list(match_list_path, full_to_abbr)

    # 从CSV读取已完成的比赛数据
    completed_matches = defaultdict(lambda: {'maps': []})
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            match_id = row['MatchId']
            map_idx = int(row['MapIdx'])

            # 只处理每个地图的第一行记录
            if map_idx > len(completed_matches[match_id]['maps']):
                scores = list(map(int, row['Scores'].split('-')))
                completed_matches[match_id]['maps'].append({
                    'name': dq(row['Map']),  # 使用双引号包裹字符串
                    'score': scores
                })

    # 构建YAML数据结构
    yaml_data = {'regular_season': [], 'playoffs': []}

    # 按顺序处理所有比赛
    for idx, match_id, teams, status, scores in all_matches:
        # 使用双引号包裹队伍缩写
        quoted_teams = [dq(team) for team in teams]

        if status == 'Completed':
            # 计算总比分
            total_score = [0, 0]
            for map_data in completed_matches[match_id]['maps']:
                if map_data['score'][0] > map_data['score'][1]:
                    total_score[0] += 1
                else:
                    total_score[1] += 1

            yaml_data['regular_season'].append({
                'match': quoted_teams + [total_score],
                'maps': completed_matches[match_id]['maps']
            })
        else:  # 未开始的比赛
            yaml_data['regular_season'].append({
                'match': quoted_teams + [None],
                'maps': None
            })

    # 使用ruamel.yaml写入YAML文件
    ryaml = YAML()
    ryaml.preserve_quotes = True  # 保留字符串的引号
    ryaml.default_flow_style = False  # 强制使用块式风格

    with open(output_yaml_path, 'w', encoding='utf-8') as f:
        f.write("# https://www.vlr.gg/event/matches/2499/vct-2025-china-stage-2/\n")
        ryaml.dump(yaml_data, f)



if __name__ == "__main__":
    # csv_to_yaml('25CN_STAGE2_team_map_perf.csv', 'results_detailed.yaml')
    # csv_to_yaml_with_coming_ryaml(
    #     '25CN_STAGE2_team_map_perf.csv',
    #     '25CN_STAGE2_match_list.yaml',
    #     'team_name.yaml',
    #     'results_detailed.yaml'
    # )
    csv_to_yaml_with_coming(
        '25CN_STAGE2_team_map_perf.csv',
        '25CN_STAGE2_match_list.yaml',
        'team_name.yaml',
        'results_detailed.yaml'
    )
