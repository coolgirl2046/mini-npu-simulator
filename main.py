"""
Mini NPU Simulator
- MAC(Multiply-Accumulate) 연산으로 패턴(Cross / X)을 판별하는 콘솔 프로그램
- 외부 라이브러리 없이 표준 라이브러리(json, time, os, re)만 사용
"""

import json
import os
import re
import time

# ------------------------------------------------------------
# 설정값
# ------------------------------------------------------------
EPSILON = 1e-9          # 두 점수 차이가 이 값보다 작으면 동점으로 본다
EPSILON_TEXT = "1e-9"   # 화면 출력용
REPEAT = 10             # 성능 측정 반복 횟수
PERF_SIZES = [3, 5, 13, 25]
DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.json")

LABEL_CROSS = "Cross"
LABEL_X = "X"
LABEL_UNDECIDED = "UNDECIDED"


# ------------------------------------------------------------
# 1. 데이터 구조: n×n 2차원 배열(리스트 안의 리스트)
# ------------------------------------------------------------
def make_matrix(n, fill=0.0):
    """n×n 크기의 2차원 배열을 만든다."""
    return [[fill for _ in range(n)] for _ in range(n)]


def get_value(matrix, row, col):
    """특정 위치(row, col)의 값을 읽는다."""
    return matrix[row][col]


def set_value(matrix, row, col, value):
    """특정 위치(row, col)에 값을 저장한다."""
    matrix[row][col] = value


def check_square(matrix):
    """정사각 2차원 숫자 배열이면 크기 N을, 아니면 오류 메시지를 반환한다.
    반환값: (N 또는 None, 오류메시지 또는 None)
    """
    if not isinstance(matrix, list) or len(matrix) == 0:
        return None, "2차원 배열이 아님"
    n = len(matrix)
    for i, row in enumerate(matrix):
        if not isinstance(row, list):
            return None, f"{i}번째 행이 배열이 아님"
        if len(row) != n:
            return None, f"{i}번째 행의 열 수({len(row)})가 행 수({n})와 다름"
        for v in row:
            # bool은 int의 하위 타입이라 따로 제외
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                return None, f"숫자가 아닌 값 포함: {v!r}"
    return n, None


# ------------------------------------------------------------
# 2. 라벨 정규화: 여러 표기를 표준 라벨(Cross / X)로 통일
# ------------------------------------------------------------
LABEL_MAP = {
    "+": LABEL_CROSS,
    "cross": LABEL_CROSS,
    "x": LABEL_X,
}


def normalize_label(raw):
    """'+', 'cross' → Cross / 'x' → X. 알 수 없는 값이면 None."""
    if not isinstance(raw, str):
        return None
    return LABEL_MAP.get(raw.strip().lower())


# ------------------------------------------------------------
# 3. MAC 연산 (반복문으로 직접 구현)
# ------------------------------------------------------------
def mac(pattern, filt):
    """같은 위치끼리 곱한 뒤 모두 더한다. (N×N번 곱셈)"""
    n = len(pattern)
    total = 0.0
    for i in range(n):
        for j in range(n):
            total += pattern[i][j] * filt[i][j]
    return total


def flatten(matrix):
    """[보너스] 2차원 배열을 길이 N²인 1차원 배열로 펼친다."""
    flat = []
    for row in matrix:
        for v in row:
            flat.append(v)
    return flat


def mac_flat(pattern_flat, filt_flat):
    """[보너스] 1차원 배열 버전 MAC. 반복문 1개로 끝난다."""
    total = 0.0
    for k in range(len(pattern_flat)):
        total += pattern_flat[k] * filt_flat[k]
    return total


# ------------------------------------------------------------
# 4. 점수 비교 정책 (epsilon)
# ------------------------------------------------------------
def judge(score_a, score_b, label_a, label_b):
    """두 점수를 비교해 더 큰 쪽의 라벨을 반환. 차이가 EPSILON 미만이면 UNDECIDED."""
    if abs(score_a - score_b) < EPSILON:
        return LABEL_UNDECIDED
    return label_a if score_a > score_b else label_b


# ------------------------------------------------------------
# 5. 성능 측정용 패턴 생성 (크기별로 같은 조건의 입력을 만들기 위함)
# ------------------------------------------------------------
def generate_cross(n):
    """가운데 행과 가운데 열이 1인 십자가 패턴."""
    m = make_matrix(n)
    mid = n // 2
    for i in range(n):
        set_value(m, mid, i, 1.0)
        set_value(m, i, mid, 1.0)
    return m


def generate_x(n):
    """두 대각선이 1인 X 패턴."""
    m = make_matrix(n)
    for i in range(n):
        set_value(m, i, i, 1.0)
        set_value(m, i, n - 1 - i, 1.0)
    return m


# ------------------------------------------------------------
# 6. 성능 측정
# ------------------------------------------------------------
def measure_ms(func, *args, repeat=REPEAT):
    """func(*args)를 repeat번 호출해 1회 평균 시간(ms)을 반환. 입출력은 포함하지 않는다."""
    func(*args)  # 워밍업 1회(측정 제외)
    start = time.perf_counter()
    for _ in range(repeat):
        func(*args)
    end = time.perf_counter()
    return (end - start) * 1000 / repeat


def print_performance(sizes):
    """크기별 MAC 평균 시간 표 출력 (2D 기본 + 1D 보너스 비교)."""
    print_section(f"성능 분석 (평균/{REPEAT}회)")
    # 한글은 화면에서 2칸을 차지해 f-string 정렬이 어긋나므로 머리글은 고정 문자열로 둔다
    print("크기          평균 시간(ms)   1D 최적화(ms)   연산 횟수(N²)")
    print("-" * 60)
    for n in sizes:
        pattern = generate_cross(n)
        filt = generate_x(n)
        t2d = measure_ms(mac, pattern, filt)
        # 펼치는 작업은 측정 구간 밖에서 미리 해 둔다(동일 입력, 동일 반복 횟수)
        pf, ff = flatten(pattern), flatten(filt)
        t1d = measure_ms(mac_flat, pf, ff)
        size_text = f"{n}x{n}"
        print(f"{size_text:<8}{t2d:>16.4f}{t1d:>16.4f}{n * n:>16}")


# ------------------------------------------------------------
# 출력 도우미
# ------------------------------------------------------------
_section_no = [0]


def reset_section():
    _section_no[0] = 0


def print_section(title):
    _section_no[0] += 1
    print()
    print("#" + "-" * 40)
    print(f"# [{_section_no[0]}] {title}")
    print("#" + "-" * 40)


# ------------------------------------------------------------
# 모드 1: 사용자 입력 (3×3)
# ------------------------------------------------------------
def read_matrix(n, title):
    """n줄을 입력받아 n×n 배열로 만든다. 형식이 틀리면 처음부터 다시 입력받는다."""
    while True:
        print(f"{title} ({n}줄 입력, 공백 구분)")
        matrix = make_matrix(n)
        ok = True
        for i in range(n):
            line = input().strip()
            parts = line.split()
            if len(parts) != n:
                print(f"입력 형식 오류: 각 줄에 {n}개의 숫자를 공백으로 구분해 입력하세요. "
                      f"({i + 1}번째 줄에 {len(parts)}개 입력됨)")
                ok = False
                break
            try:
                for j, p in enumerate(parts):
                    set_value(matrix, i, j, float(p))
            except ValueError:
                print(f"입력 형식 오류: 숫자가 아닌 값이 있습니다. ({i + 1}번째 줄: {line})")
                ok = False
                break
        if ok:
            return matrix
        print("다시 입력해 주세요.\n")


def print_matrix(matrix):
    for row in matrix:
        print("  " + " ".join(f"{v:g}" for v in row))


def run_user_mode():
    n = 3
    print_section("필터 입력")
    filter_a = read_matrix(n, "필터 A")
    print()
    filter_b = read_matrix(n, "필터 B")

    print("\n저장된 필터 A:")
    print_matrix(filter_a)
    print("저장된 필터 B:")
    print_matrix(filter_b)

    print_section("패턴 입력")
    pattern = read_matrix(n, "패턴")

    # 연산 구간: 입출력 없이 MAC만
    score_a = mac(pattern, filter_a)
    score_b = mac(pattern, filter_b)
    avg_ms = measure_ms(lambda: (mac(pattern, filter_a), mac(pattern, filter_b)))
    result = judge(score_a, score_b, "A", "B")

    print_section("MAC 결과")
    print(f"A 점수: {score_a!r}")
    print(f"B 점수: {score_b!r}")
    print(f"연산 시간(A+B 1회 평균/{REPEAT}회): {avg_ms:.4f} ms")
    if result == LABEL_UNDECIDED:
        print(f"판정: 판정 불가 (|A-B| < {EPSILON_TEXT})")
    else:
        print(f"판정: {result}")

    print_performance([3])


# ------------------------------------------------------------
# 모드 2: data.json 분석
# ------------------------------------------------------------
def load_json(path):
    """JSON 파일을 읽는다. 실패하면 (None, 오류메시지)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f), None
    except FileNotFoundError:
        return None, f"파일을 찾을 수 없습니다: {path}"
    except json.JSONDecodeError as e:
        return None, f"JSON 형식 오류: {e}"


def load_filters(raw_filters):
    """filters 항목을 읽어 {N: {'Cross': 배열, 'X': 배열}} 형태로 정리한다."""
    filters = {}
    if not isinstance(raw_filters, dict):
        print("✗ filters 항목이 없거나 형식이 올바르지 않습니다.")
        return filters

    for key, value in raw_filters.items():
        m = re.fullmatch(r"size_(\d+)", str(key))
        if not m:
            print(f"✗ {key}: 키 형식이 size_N이 아님 (건너뜀)")
            continue
        n = int(m.group(1))
        if not isinstance(value, dict):
            print(f"✗ {key}: 필터 묶음 형식 오류 (건너뜀)")
            continue

        bucket = {}
        for fkey, matrix in value.items():
            label = normalize_label(fkey)          # 'cross' → Cross, 'x' → X
            if label is None:
                print(f"✗ {key}.{fkey}: 알 수 없는 필터 이름 (건너뜀)")
                continue
            size, err = check_square(matrix)
            if err:
                print(f"✗ {key}.{fkey}: {err} (건너뜀)")
                continue
            if size != n:
                print(f"✗ {key}.{fkey}: 키는 {n}인데 실제 크기는 {size}x{size} (건너뜀)")
                continue
            bucket[label] = matrix

        if LABEL_CROSS in bucket and LABEL_X in bucket:
            filters[n] = bucket
            print(f"✓ {key:<8} 필터 로드 완료 (Cross, X)")
        else:
            print(f"✗ {key:<8} 필터 로드 실패 (Cross/X 중 누락: {sorted(bucket)})")
    return filters


def analyze_pattern(key, case, filters):
    """패턴 1건을 분석해 (PASS 여부, 실패 사유)를 반환한다. 어떤 오류도 예외로 끝내지 않는다."""
    print(f"--- {key} ---")

    def fail(reason):
        print(f"FAIL: {reason}")
        return False, reason

    m = re.fullmatch(r"size_(\d+)_(\d+)", str(key))
    if not m:
        return fail("키 형식 오류 (size_N_idx 아님)")
    n = int(m.group(1))

    if n not in filters:
        return fail(f"size_{n} 필터 없음")
    if not isinstance(case, dict):
        return fail("케이스 형식 오류 (객체 아님)")

    expected = normalize_label(case.get("expected"))
    if expected is None:
        return fail(f"expected 라벨 해석 불가: {case.get('expected')!r}")

    size, err = check_square(case.get("input"))
    if err:
        return fail(f"input 형식 오류: {err}")
    if size != n:
        return fail(f"크기 불일치: 필터 {n}x{n}, 패턴 {size}x{size}")

    pattern = case["input"]
    score_cross = mac(pattern, filters[n][LABEL_CROSS])
    score_x = mac(pattern, filters[n][LABEL_X])
    result = judge(score_cross, score_x, LABEL_CROSS, LABEL_X)

    print(f"Cross 점수: {score_cross!r}")
    print(f"X 점수: {score_x!r}")

    if result == LABEL_UNDECIDED:
        print(f"판정: {result} | expected: {expected} | FAIL (동점 규칙)")
        return False, f"동점(UNDECIDED, |차이| < {EPSILON_TEXT}) 규칙에 따라 FAIL"
    if result == expected:
        print(f"판정: {result} | expected: {expected} | PASS")
        return True, None
    print(f"판정: {result} | expected: {expected} | FAIL")
    return False, f"판정 {result} ≠ expected {expected}"


def run_json_mode(path=DATA_FILE):
    print_section("필터 로드")
    data, err = load_json(path)
    if err:
        print(f"✗ {err}")
        return
    if not isinstance(data, dict):
        print("✗ data.json 최상위가 객체({})가 아닙니다.")
        return

    filters = load_filters(data.get("filters"))

    print_section("패턴 분석 (라벨 정규화 적용)")
    patterns = data.get("patterns")
    if not isinstance(patterns, dict):
        print("✗ patterns 항목이 없거나 형식이 올바르지 않습니다.")
        patterns = {}

    total, passed, failures = 0, 0, []
    for key, case in patterns.items():
        total += 1
        try:
            ok, reason = analyze_pattern(key, case, filters)
        except Exception as e:  # 예상 못한 오류도 케이스 단위 FAIL로 처리
            ok, reason = False, f"예상치 못한 오류: {e}"
            print(f"FAIL: {reason}")
        if ok:
            passed += 1
        else:
            failures.append((key, reason))

    print_performance(PERF_SIZES)

    print_section("결과 요약")
    print(f"총 테스트: {total}개")
    print(f"통과: {passed}개")
    print(f"실패: {len(failures)}개")
    if failures:
        print("\n실패 케이스:")
        for key, reason in failures:
            print(f"- {key}: {reason}")


# ------------------------------------------------------------
# 메인
# ------------------------------------------------------------
def main():
    print("=== Mini NPU Simulator ===")
    while True:
        print("\n[모드 선택]")
        print("1. 사용자 입력 (3x3)")
        print("2. data.json 분석")
        choice = input("선택: ").strip()
        if choice in ("1", "2"):
            break
        print("1 또는 2를 입력하세요.")

    reset_section()
    if choice == "1":
        run_user_mode()
    else:
        run_json_mode()


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\n프로그램을 종료합니다.")
