import csv
import yaml
from collections import defaultdict

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
        yaml.dump(yaml_data, f, sort_keys=False, allow_unicode=True, default_flow_style=None)

# 使用示例
csv_to_yaml('25CN_STAGE2_team_map_perf.csv', 'results_detailed.yaml')