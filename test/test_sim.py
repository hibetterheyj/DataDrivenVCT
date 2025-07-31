import os
import sys
import yaml
import random
from pathlib import Path
import pytest
from typing import List, Dict, Tuple, Any

# 将项目根目录添加到Python路径
sys.path.append(str(Path(__file__).parent.parent))
from sim import (
    load_yaml,
    load_group_teams,
    load_initial_pts,
    get_bo_score_probs,
    simulate_match,
    load_map_pool,
    _parse_score_from_result,
    play_regular_season,
    compare_teams,
    get_qualified,
    play_playoffs
)

# 测试数据目录
TEST_DATA_DIR = Path(__file__).parent / "test_data"

@pytest.fixture(autouse=True)
def setup_test_data():
    """创建测试所需的临时目录和文件"""
    TEST_DATA_DIR.mkdir(exist_ok=True)
    (TEST_DATA_DIR / "cn").mkdir(exist_ok=True)

    # 创建测试用的分组文件
    groups_data = {
        "Alpha": ["TeamA", "TeamB", "TeamC", "TeamD", "TeamE", "TeamF"],
        "Omega": ["TeamG", "TeamH", "TeamI", "TeamJ", "TeamK", "TeamL"]
    }
    with open(TEST_DATA_DIR / "cn" / "groups.yaml", "w") as f:
        yaml.dump(groups_data, f)

    # 创建测试用的初始积分文件
    initial_pts_data = {
        "TeamA": 10, "TeamB": 8, "TeamC": 6, "TeamD": 4,
        "TeamE": 2, "TeamF": 0, "TeamG": 10, "TeamH": 8,
        "TeamI": 6, "TeamJ": 4, "TeamK": 2, "TeamL": 0
    }
    with open(TEST_DATA_DIR / "cn" / "initial_pts.yaml", "w") as f:
        yaml.dump(initial_pts_data, f)

    # 创建测试用的地图池文件
    map_pool_data = ["Ascent", "Bind", "Haven", "Icebox"]
    with open(TEST_DATA_DIR / "map_pool.yaml", "w") as f:
        yaml.dump(map_pool_data, f)

    yield

    # 清理测试数据（可选）
    # import shutil
    # shutil.rmtree(TEST_DATA_DIR)

def test_load_yaml():
    """测试YAML文件加载功能"""
    test_file = TEST_DATA_DIR / "test.yaml"
    test_data = {"key": "value", "list": [1, 2, 3]}

    with open(test_file, "w") as f:
        yaml.dump(test_data, f)

    loaded_data = load_yaml(str(test_file))
    assert loaded_data == test_data

def test_load_group_teams():
    """测试加载分组功能"""
    alpha, omega = load_group_teams(str(TEST_DATA_DIR), "cn")
    assert len(alpha) == 6
    assert len(omega) == 6
    assert "TeamA" in alpha
    assert "TeamG" in omega

def test_load_initial_pts():
    """测试加载初始积分功能"""
    initial_pts = load_initial_pts(str(TEST_DATA_DIR), "cn")
    assert len(initial_pts) == 12
    assert initial_pts["TeamA"] == 10
    assert initial_pts["TeamL"] == 0

def test_get_bo_score_probs():
    """测试BO赛制比分概率计算"""
    # 测试BO3
    bo3_probs = get_bo_score_probs(3, 0.5)
    assert set(bo3_probs.keys()) == {"2:0", "0:2", "2:1", "1:2"}
    assert sum(bo3_probs.values()) == pytest.approx(1.0)   # 使用pytest近似比较

    # 测试BO5
    bo5_probs = get_bo_score_probs(5, 0.5)
    assert set(bo5_probs.keys()) == {"3:0", "0:3", "3:1", "1:3", "3:2", "2:3"}
    assert sum(bo5_probs.values()) == pytest.approx(1.0)

    # 测试无效BO值
    with pytest.raises(ValueError):
        get_bo_score_probs(4, 0.5)

def test_simulate_match():
    """测试模拟比赛功能"""
    # 固定随机种子以确保结果可预测
    random.seed(42)

    # 测试BO3
    winner, score = simulate_match("TeamA", "TeamB", 3, 0.5)
    assert winner in ["TeamA", "TeamB"]
    assert len(score) == 2
    assert max(score) == 2  # BO3需要赢2局

    # 测试BO5
    winner, score = simulate_match("TeamA", "TeamB", 5, 0.5)
    assert max(score) == 3  # BO5需要赢3局

    # 测试无效参数
    with pytest.raises(ValueError):
        simulate_match("TeamA", "TeamB", 4, 0.5)

    with pytest.raises(ValueError):
        simulate_match("TeamA", "TeamB", 3, 1.5)

def test_load_map_pool():
    """测试加载地图池功能"""
    map_pool = load_map_pool(str(TEST_DATA_DIR))
    assert len(map_pool) == 4
    assert "Ascent" in map_pool

    # 测试文件不存在的情况（应返回默认地图池）
    non_existent_dir = TEST_DATA_DIR / "nonexistent"
    default_map_pool = load_map_pool(str(non_existent_dir))
    assert len(default_map_pool) == 7  # 默认地图池有7张地图

def test_parse_score_from_result():
    """测试比分解析功能"""
    # 测试地图级比分
    map_scores = [[13, 11], [8, 13], [13, 10]]
    team1_maps, team2_maps = _parse_score_from_result(map_scores)
    assert team1_maps == 2
    assert team2_maps == 1

    # 测试传统比分
    traditional_score = [2, 1]
    team1_maps, team2_maps = _parse_score_from_result(traditional_score)
    assert team1_maps == 2
    assert team2_maps == 1

"""测试常规赛模拟功能"""
def test_play_regular_season():
    """测试常规赛模拟功能"""
    group = ["TeamA", "TeamB", "TeamC", "TeamD"]
    random.seed(42)

    pts, win_loss, map_diff, team_stats, head_to_head, match_records = play_regular_season(
        group, use_real_data=False, map_based=False
    )

    # 检查每个队伍都有积分
    assert set(pts.keys()) == set(group)

    # 检查比赛数量：4支队伍应进行6场比赛
    assert len(match_records) == 6

    # 检查胜负记录是否合理（修正解析逻辑）
    for team in group:
        # 分割胜场和负场部分（格式为"X胜-Y负"）
        win_part, loss_part = win_loss[team].split("胜-")
        # 提取数字部分并转换为整数
        wins = int(win_part)
        losses = int(loss_part.replace("负", ""))  # 移除"负"字后转换
        assert wins + losses == 3  # 每个队打3场比赛（4支队伍每队交手3次）

def test_compare_teams():
    """测试队伍比较功能"""
    team_stats = {
        "TeamA": {"wins": 5, "maps_won": 10, "maps_lost": 5},
        "TeamB": {"wins": 4, "maps_won": 12, "maps_lost": 6},
        "TeamC": {"wins": 5, "maps_won": 9, "maps_lost": 6},
        "TeamD": {"wins": 5, "maps_won": 10, "maps_lost": 5}
    }

    head_to_head = {
        "TeamA": {"TeamD": 0},
        "TeamD": {"TeamA": 1},
        "TeamB": {},
        "TeamC": {}
    }

    # 测试胜场不同的情况
    assert compare_teams("TeamA", "TeamB", team_stats, head_to_head) is True

    # 测试胜场相同但地图差不同的情况
    assert compare_teams("TeamA", "TeamC", team_stats, head_to_head) is True

    # 测试胜场和地图差都相同但交锋记录不同的情况
    assert compare_teams("TeamA", "TeamD", team_stats, head_to_head) is False

def test_get_qualified():
    """测试晋级队伍选择功能"""
    group = ["TeamA", "TeamB", "TeamC", "TeamD", "TeamE", "TeamF"]

    # 构造明显的排名情况
    team_stats = {
        "TeamA": {"wins": 5, "maps_won": 15, "maps_lost": 3},
        "TeamB": {"wins": 4, "maps_won": 12, "maps_lost": 6},
        "TeamC": {"wins": 3, "maps_won": 10, "maps_lost": 8},
        "TeamD": {"wins": 3, "maps_won": 9, "maps_lost": 9},
        "TeamE": {"wins": 2, "maps_won": 8, "maps_lost": 10},
        "TeamF": {"wins": 1, "maps_won": 5, "maps_lost": 13}
    }

    pts = {t: team_stats[t]["wins"] for t in group}
    win_loss = {t: f"{team_stats[t]['wins']}胜-{5-team_stats[t]['wins']}负" for t in group}
    map_diff = {t: f"{team_stats[t]['maps_won']-team_stats[t]['maps_lost']}" for t in group}
    head_to_head = {t: {} for t in group}

    qualified = get_qualified(group, pts, win_loss, map_diff, team_stats, head_to_head)

    # 应该有4支队伍晋级
    assert len(qualified) == 4

    # 前两名应该是TeamA和TeamB
    assert qualified[0] == "TeamA"
    assert qualified[1] == "TeamB"

def test_play_playoffs():
    """测试季后赛模拟功能"""
    qualified_a = ["TeamA", "TeamB", "TeamC", "TeamD"]  # Alpha组前4
    qualified_b = ["TeamG", "TeamH", "TeamI", "TeamJ"]  # Omega组前4
    initial_pts = {t: 0 for t in qualified_a + qualified_b}
    regular_pts = {t: 10 - i*2 for i, t in enumerate(qualified_a + qualified_b)}

    random.seed(42)
    playoff_results = play_playoffs(
        qualified_a, qualified_b, initial_pts, regular_pts,
        use_real_data=False, map_based=False
    )

    # 检查结果包含必要的键
    assert "champion" in playoff_results
    assert "runner_up" in playoff_results
    assert "third_place" in playoff_results
    assert "fourth_place" in playoff_results
    assert "champions_slots" in playoff_results

    # 检查冠军赛参赛队伍数量
    assert len(playoff_results["champions_slots"]) == 4

    # 检查季后赛轮次数据
    assert len(playoff_results["rounds"]) == 12  # M1到M12