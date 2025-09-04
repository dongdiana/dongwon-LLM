import pandas as pd

def preprocessing_population_data(data):
    """인구조사 데이터 전처리"""
    gender = {1: '남성', 2: '여성'}
    df = data.copy().drop(columns=['행정구역시군구코드', '행정구역시군구코드', '동읍면부_구분코드', '조사구특성코드', '성별코드',
                                '만연령', '가구주관계코드', '교육정도코드', '수학구분코드', '종사상지위코드', '근로장소코드', '초혼연령코드'])
    df['성별'] = data['성별코드'].map(gender)
    df['연령대'] = data['만연령'].apply(lambda x: (x//10)*10 if x < 80 else 70)

    df['가구원수'] = df.groupby(['가구일련번호'])["가구원번호"].transform("max")
    df['가구원수'] = df['가구원수'].apply(lambda x: f"{x}인가구" if x < 3 else "3인가구 이상")

    df.drop(columns=['가구일련번호','행정구역시도코드', '가구원번호','경제활동상태코드',
                 '산업대분류코드','직업대분류코드', '인구가중값'], inplace=True)
    
    df = df[(df['연령대'] > 10)].reset_index(drop=True)
    df['연령대'] = df['연령대'].apply(lambda x: f"{x}대" if x < 70 else "70대 이상")

    df.to_csv('./data/population/2020_2%_인구사항_전처리.csv', index=False, encoding='utf-8')
    
    return df

def preprocessing_household_data(data_2):
    """가구 자료 전처리"""
    share_cols = ['가구원수', '소득구간코드','가계지출금액','가계지출_소비지출금액','가구일련번호']
    share_df = data_2[share_cols]
    df_base = data_2.iloc[:,3:5]
    df_base = pd.concat([df_base, share_df], axis=1)

    df_base.columns = ['성별','연령','가구원수', '소득구간코드','가계지출금액','가계지출_소비지출금액','가구일련번호']

    result = df_base.copy()
    for i in range(2,10):
        temp = data_2.iloc[:,i*2+1:i*2+3]
        temp.columns = ['성별','연령']
        temp = pd.concat([temp, share_df], axis=1)
        temp.dropna(how='any', inplace=True)
        temp.reset_index(inplace=True, drop=True)
        result = pd.concat([result, temp], axis=0).reset_index(drop=True)
    
    gender = {1:'남자', 2:'여자'}
    income = {1:'저소득(월 300만원 미만)', 2:'저소득(월 300만원 미만)', 3:'저소득(월 300만원 미만)',
            4:'중간소득(월 300~700만원 미만)', 5:'중간소득(월 300~700만원 미만)', 6:'중간소득(월 300~700만원 미만)', 7:'중간소득(월 300~700만원 미만)',
            8:'고소득(월 700만원 이상)'}
    
    result['성별'] = result['성별'].map(gender)
    result['소득구간코드'] = result['소득구간코드'].map(income)

    result['연령'] = result['연령'].apply(lambda x: f"{int(x)//10 * 10}대" if x < 70 else "70대 이상")
    result['가구원수'] = result['가구원수'].apply(lambda x: f"{int(x)}인가구" if int(x) < 3 else "3인가구 이상")

    result.columns = ['성별','연령대','가구원수', '소득구간','가계지출금액','가계지출_소비지출금액','가구일련번호']

    result = result[(result['연령대'] != '0대') & (result['연령대'] != '10대')]
    result.to_csv('./data/population/표본추출대상집단.csv', index=False, encoding='utf-8')
    
    return result

def sampling_wt_lb(df, n_samples=1000, min_count=10):
    """조건을 만족할 때까지 무작위 샘플링"""
    seed = 0
    while True:
        df_sample = df.sample(n=n_samples, random_state=seed)
        if (df_sample['성별'].value_counts().min() >= min_count and
            df_sample['연령대'].value_counts().min() >= min_count and
            df_sample['가구원수'].value_counts().min() >= min_count and
            df_sample['소득구간'].value_counts().min() >= min_count):
            break
        seed += 1
    return seed, n_samples, df_sample

def save_results(sample: pd.DataFrame, n_samples):
    """샘플링 결과 저장"""
    sample.reset_index(inplace=True)
    instances = sample.iloc[:,:-3]
    save_path = f'./data/persona.csv'
    instances.to_csv(save_path, index=False, encoding='utf-8')


if __name__ == "__main__":
    data = pd.read_csv('./data/population/2020_2%_인구사항.csv', encoding='cp949')
    processed_pop = preprocessing_population_data(data)

    data_2 = pd.read_csv('./data/population/2024_연간자료(지출, 신항목분류, 2019~) - 전체가구_1인이상_20250828_24357.csv', encoding='cp949')
    processed_house = preprocessing_household_data(data_2)

    seed, n_samples, sample = sampling_wt_lb(processed_house, n_samples=1000, min_count=50)
    print(f"샘플링 완료: seed={seed}, shape={sample.shape}")
    save_results(sample, n_samples)