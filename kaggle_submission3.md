# 데이터 로드


```python
import pandas as pd
import seaborn as sns

df=pd.read_csv('c:\\data\\train.csv')

pd.set_option('display.max_columns', 15 )
df

#df['Age'].mean() # 평균 29.69 , 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>886</th>
      <td>887</td>
      <td>0</td>
      <td>2</td>
      <td>Montvila, Rev. Juozas</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>211536</td>
      <td>13.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>887</th>
      <td>888</td>
      <td>1</td>
      <td>1</td>
      <td>Graham, Miss. Margaret Edith</td>
      <td>female</td>
      <td>19.0</td>
      <td>0</td>
      <td>0</td>
      <td>112053</td>
      <td>30.0000</td>
      <td>B42</td>
      <td>S</td>
    </tr>
    <tr>
      <th>888</th>
      <td>889</td>
      <td>0</td>
      <td>3</td>
      <td>Johnston, Miss. Catherine Helen "Carrie"</td>
      <td>female</td>
      <td>NaN</td>
      <td>1</td>
      <td>2</td>
      <td>W./C. 6607</td>
      <td>23.4500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>889</th>
      <td>890</td>
      <td>1</td>
      <td>1</td>
      <td>Behr, Mr. Karl Howell</td>
      <td>male</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>111369</td>
      <td>30.0000</td>
      <td>C148</td>
      <td>C</td>
    </tr>
    <tr>
      <th>890</th>
      <td>891</td>
      <td>0</td>
      <td>3</td>
      <td>Dooley, Mr. Patrick</td>
      <td>male</td>
      <td>32.0</td>
      <td>0</td>
      <td>0</td>
      <td>370376</td>
      <td>7.7500</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
  </tbody>
</table>
<p>891 rows × 12 columns</p>
</div>



# 여자-아이 파생변수 추가 


```python
mask= (df.Age <10) | (df.Sex=='female')
mask.astype(int)
df['women_child']=mask.astype(int)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>women_child</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>886</th>
      <td>887</td>
      <td>0</td>
      <td>2</td>
      <td>Montvila, Rev. Juozas</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>211536</td>
      <td>13.0000</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>887</th>
      <td>888</td>
      <td>1</td>
      <td>1</td>
      <td>Graham, Miss. Margaret Edith</td>
      <td>female</td>
      <td>19.0</td>
      <td>0</td>
      <td>0</td>
      <td>112053</td>
      <td>30.0000</td>
      <td>B42</td>
      <td>S</td>
      <td>1</td>
    </tr>
    <tr>
      <th>888</th>
      <td>889</td>
      <td>0</td>
      <td>3</td>
      <td>Johnston, Miss. Catherine Helen "Carrie"</td>
      <td>female</td>
      <td>NaN</td>
      <td>1</td>
      <td>2</td>
      <td>W./C. 6607</td>
      <td>23.4500</td>
      <td>NaN</td>
      <td>S</td>
      <td>1</td>
    </tr>
    <tr>
      <th>889</th>
      <td>890</td>
      <td>1</td>
      <td>1</td>
      <td>Behr, Mr. Karl Howell</td>
      <td>male</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>111369</td>
      <td>30.0000</td>
      <td>C148</td>
      <td>C</td>
      <td>0</td>
    </tr>
    <tr>
      <th>890</th>
      <td>891</td>
      <td>0</td>
      <td>3</td>
      <td>Dooley, Mr. Patrick</td>
      <td>male</td>
      <td>32.0</td>
      <td>0</td>
      <td>0</td>
      <td>370376</td>
      <td>7.7500</td>
      <td>NaN</td>
      <td>Q</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>891 rows × 13 columns</p>
</div>



# 데이터 전처리(중복 삭제, 결측치 대체)


```python
# 중복되는 deck, embarked_town, class 행 삭제하기(embarked만 남김)
rdf = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1 ) 

#embared 결측치를 최빈값으로 대체하기
most_freq = rdf['Embarked'].value_counts(dropna=True).idxmax()

# 나이 결측치 평균값으로 대체하기
mean_age = rdf['Age'].mean()
mean_age
rdf['Age'].fillna(mean_age, inplace=True)
rdf
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>women_child</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>22.000000</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>38.000000</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>C</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>26.000000</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>S</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>35.000000</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>S</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>35.000000</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>886</th>
      <td>0</td>
      <td>2</td>
      <td>male</td>
      <td>27.000000</td>
      <td>0</td>
      <td>0</td>
      <td>13.0000</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>887</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>19.000000</td>
      <td>0</td>
      <td>0</td>
      <td>30.0000</td>
      <td>S</td>
      <td>1</td>
    </tr>
    <tr>
      <th>888</th>
      <td>0</td>
      <td>3</td>
      <td>female</td>
      <td>29.699118</td>
      <td>1</td>
      <td>2</td>
      <td>23.4500</td>
      <td>S</td>
      <td>1</td>
    </tr>
    <tr>
      <th>889</th>
      <td>1</td>
      <td>1</td>
      <td>male</td>
      <td>26.000000</td>
      <td>0</td>
      <td>0</td>
      <td>30.0000</td>
      <td>C</td>
      <td>0</td>
    </tr>
    <tr>
      <th>890</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>32.000000</td>
      <td>0</td>
      <td>0</td>
      <td>7.7500</td>
      <td>Q</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>891 rows × 9 columns</p>
</div>



# fare 이상치 제거


```python
local_std = rdf.Fare.std()*5
local_std

rdf=rdf[:][rdf['Fare'] < local_std]
rdf # 891 ->862건
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>women_child</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>22.000000</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>38.000000</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>C</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>26.000000</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>S</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>35.000000</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>S</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>35.000000</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>886</th>
      <td>0</td>
      <td>2</td>
      <td>male</td>
      <td>27.000000</td>
      <td>0</td>
      <td>0</td>
      <td>13.0000</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>887</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>19.000000</td>
      <td>0</td>
      <td>0</td>
      <td>30.0000</td>
      <td>S</td>
      <td>1</td>
    </tr>
    <tr>
      <th>888</th>
      <td>0</td>
      <td>3</td>
      <td>female</td>
      <td>29.699118</td>
      <td>1</td>
      <td>2</td>
      <td>23.4500</td>
      <td>S</td>
      <td>1</td>
    </tr>
    <tr>
      <th>889</th>
      <td>1</td>
      <td>1</td>
      <td>male</td>
      <td>26.000000</td>
      <td>0</td>
      <td>0</td>
      <td>30.0000</td>
      <td>C</td>
      <td>0</td>
    </tr>
    <tr>
      <th>890</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>32.000000</td>
      <td>0</td>
      <td>0</td>
      <td>7.7500</td>
      <td>Q</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>882 rows × 9 columns</p>
</div>



# 범주형 -> 숫자형 데이터로 변환하기 


```python

ndf=rdf[['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'women_child']]
ndf

#성별 인코딩
gender=pd.get_dummies(ndf['Sex'])
ndf=pd.concat([ndf, gender], axis=1)

# embared 인코딩
onehot_embarked=pd.get_dummies(ndf['Embarked'])
ndf=pd.concat([ndf, onehot_embarked], axis=1)
ndf.drop(['Sex', 'Embarked'], axis=1, inplace=True)
ndf
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>Fare</th>
      <th>women_child</th>
      <th>female</th>
      <th>male</th>
      <th>C</th>
      <th>Q</th>
      <th>S</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>22.000000</td>
      <td>7.2500</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>38.000000</td>
      <td>71.2833</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>26.000000</td>
      <td>7.9250</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>35.000000</td>
      <td>53.1000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>35.000000</td>
      <td>8.0500</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>886</th>
      <td>0</td>
      <td>2</td>
      <td>27.000000</td>
      <td>13.0000</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>887</th>
      <td>1</td>
      <td>1</td>
      <td>19.000000</td>
      <td>30.0000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>888</th>
      <td>0</td>
      <td>3</td>
      <td>29.699118</td>
      <td>23.4500</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>889</th>
      <td>1</td>
      <td>1</td>
      <td>26.000000</td>
      <td>30.0000</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>890</th>
      <td>0</td>
      <td>3</td>
      <td>32.000000</td>
      <td>7.7500</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>882 rows × 10 columns</p>
</div>



# 독립변수와 종속변수 


```python
x=ndf[['Pclass', 'Age', 'Fare', 'women_child', 'female', 'male', 'C', 'Q', 'S']]
y=ndf['Survived']      
x      
      
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pclass</th>
      <th>Age</th>
      <th>Fare</th>
      <th>women_child</th>
      <th>female</th>
      <th>male</th>
      <th>C</th>
      <th>Q</th>
      <th>S</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>22.000000</td>
      <td>7.2500</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>38.000000</td>
      <td>71.2833</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>26.000000</td>
      <td>7.9250</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>35.000000</td>
      <td>53.1000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>35.000000</td>
      <td>8.0500</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>886</th>
      <td>2</td>
      <td>27.000000</td>
      <td>13.0000</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>887</th>
      <td>1</td>
      <td>19.000000</td>
      <td>30.0000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>888</th>
      <td>3</td>
      <td>29.699118</td>
      <td>23.4500</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>889</th>
      <td>1</td>
      <td>26.000000</td>
      <td>30.0000</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>890</th>
      <td>3</td>
      <td>32.000000</td>
      <td>7.7500</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>882 rows × 9 columns</p>
</div>



# 독립변수 정규화 


```python
from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(x).transform(x)
X
```




    array([[ 8.17584051e-01, -5.92120506e-01, -5.95167208e-01, ...,
            -4.76134178e-01, -3.09276856e-01,  6.14918694e-01],
           [-1.59415218e+00,  6.39799109e-01,  1.15683863e+00, ...,
             2.10024831e+00, -3.09276856e-01, -1.62623126e+00],
           [ 8.17584051e-01, -2.84140603e-01, -5.76698630e-01, ...,
            -4.76134178e-01, -3.09276856e-01,  6.14918694e-01],
           ...,
           [ 8.17584051e-01,  6.72871593e-04, -1.51921356e-01, ...,
            -4.76134178e-01, -3.09276856e-01,  6.14918694e-01],
           [-1.59415218e+00, -2.84140603e-01,  2.72922451e-02, ...,
             2.10024831e+00, -3.09276856e-01, -1.62623126e+00],
           [ 8.17584051e-01,  1.77829253e-01, -5.81486780e-01, ...,
            -4.76134178e-01,  3.23334895e+00, -1.62623126e+00]])



# 테스트 데이터도 동일하게 만들어주기 


```python
x_ktest = pd.read_csv("c:\\data\\test.csv")
mask4 = (x_ktest.Age<10) | (x_ktest.Sex=='female') # | (또는)
x_ktest['women_child']=mask4.astype(int)
print(x_ktest)
```

         PassengerId  Pclass                                          Name  \
    0            892       3                              Kelly, Mr. James   
    1            893       3              Wilkes, Mrs. James (Ellen Needs)   
    2            894       2                     Myles, Mr. Thomas Francis   
    3            895       3                              Wirz, Mr. Albert   
    4            896       3  Hirvonen, Mrs. Alexander (Helga E Lindqvist)   
    ..           ...     ...                                           ...   
    413         1305       3                            Spector, Mr. Woolf   
    414         1306       1                  Oliva y Ocana, Dona. Fermina   
    415         1307       3                  Saether, Mr. Simon Sivertsen   
    416         1308       3                           Ware, Mr. Frederick   
    417         1309       3                      Peter, Master. Michael J   
    
            Sex   Age  SibSp  Parch              Ticket      Fare Cabin Embarked  \
    0      male  34.5      0      0              330911    7.8292   NaN        Q   
    1    female  47.0      1      0              363272    7.0000   NaN        S   
    2      male  62.0      0      0              240276    9.6875   NaN        Q   
    3      male  27.0      0      0              315154    8.6625   NaN        S   
    4    female  22.0      1      1             3101298   12.2875   NaN        S   
    ..      ...   ...    ...    ...                 ...       ...   ...      ...   
    413    male   NaN      0      0           A.5. 3236    8.0500   NaN        S   
    414  female  39.0      0      0            PC 17758  108.9000  C105        C   
    415    male  38.5      0      0  SOTON/O.Q. 3101262    7.2500   NaN        S   
    416    male   NaN      0      0              359309    8.0500   NaN        S   
    417    male   NaN      1      1                2668   22.3583   NaN        C   
    
         women_child  
    0              0  
    1              1  
    2              0  
    3              0  
    4              1  
    ..           ...  
    413            0  
    414            1  
    415            0  
    416            0  
    417            0  
    
    [418 rows x 12 columns]
    

# 테스트 데이터 필요없는 컬럼 삭제 


```python
rdf_x_ktest = x_ktest.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1 )
rdf_x_ktest


# 나이 결측치 평균값으로 대체하기
mean_age =rdf_x_ktest['Age'].mean()
mean_age
rdf_x_ktest['Age'].fillna(mean_age, inplace=True)
rdf_x_ktest
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>women_child</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>male</td>
      <td>34.50000</td>
      <td>0</td>
      <td>0</td>
      <td>7.8292</td>
      <td>Q</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>female</td>
      <td>47.00000</td>
      <td>1</td>
      <td>0</td>
      <td>7.0000</td>
      <td>S</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>male</td>
      <td>62.00000</td>
      <td>0</td>
      <td>0</td>
      <td>9.6875</td>
      <td>Q</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>male</td>
      <td>27.00000</td>
      <td>0</td>
      <td>0</td>
      <td>8.6625</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>female</td>
      <td>22.00000</td>
      <td>1</td>
      <td>1</td>
      <td>12.2875</td>
      <td>S</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>413</th>
      <td>3</td>
      <td>male</td>
      <td>30.27259</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>414</th>
      <td>1</td>
      <td>female</td>
      <td>39.00000</td>
      <td>0</td>
      <td>0</td>
      <td>108.9000</td>
      <td>C</td>
      <td>1</td>
    </tr>
    <tr>
      <th>415</th>
      <td>3</td>
      <td>male</td>
      <td>38.50000</td>
      <td>0</td>
      <td>0</td>
      <td>7.2500</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>416</th>
      <td>3</td>
      <td>male</td>
      <td>30.27259</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>417</th>
      <td>3</td>
      <td>male</td>
      <td>30.27259</td>
      <td>1</td>
      <td>1</td>
      <td>22.3583</td>
      <td>C</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>418 rows × 8 columns</p>
</div>




```python
#embared 결측치를 최빈값으로 대체하기
most_freq = rdf_x_ktest['Embarked'].value_counts(dropna=True).idxmax()


fillfare = rdf_x_ktest['Fare'].value_counts().idxmax()
rdf_x_ktest['Fare'].fillna(fillfare, inplace=True) 

```


```python
ndf_x_ktest=rdf_x_ktest
ndf_x_ktest

#성별 인코딩
gender=pd.get_dummies(ndf_x_ktest['Sex'])
ndf_x_ktest=pd.concat([ndf_x_ktest, gender], axis=1)

# embared 인코딩
onehot_embarked=pd.get_dummies(ndf_x_ktest['Embarked'])
ndf_x_ktest=pd.concat([ndf_x_ktest, onehot_embarked], axis=1)
ndf_x_ktest.drop(['Sex', 'Embarked'], axis=1, inplace=True)
ndf_x_ktest
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>women_child</th>
      <th>female</th>
      <th>male</th>
      <th>C</th>
      <th>Q</th>
      <th>S</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>34.50000</td>
      <td>0</td>
      <td>0</td>
      <td>7.8292</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>47.00000</td>
      <td>1</td>
      <td>0</td>
      <td>7.0000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>62.00000</td>
      <td>0</td>
      <td>0</td>
      <td>9.6875</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>27.00000</td>
      <td>0</td>
      <td>0</td>
      <td>8.6625</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>22.00000</td>
      <td>1</td>
      <td>1</td>
      <td>12.2875</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>413</th>
      <td>3</td>
      <td>30.27259</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>414</th>
      <td>1</td>
      <td>39.00000</td>
      <td>0</td>
      <td>0</td>
      <td>108.9000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>415</th>
      <td>3</td>
      <td>38.50000</td>
      <td>0</td>
      <td>0</td>
      <td>7.2500</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>416</th>
      <td>3</td>
      <td>30.27259</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>417</th>
      <td>3</td>
      <td>30.27259</td>
      <td>1</td>
      <td>1</td>
      <td>22.3583</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>418 rows × 11 columns</p>
</div>




```python
x=ndf_x_ktest[['Pclass', 'Age', 'Fare', 'women_child', 'female', 'male', 'C', 'Q', 'S']]

x
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pclass</th>
      <th>Age</th>
      <th>Fare</th>
      <th>women_child</th>
      <th>female</th>
      <th>male</th>
      <th>C</th>
      <th>Q</th>
      <th>S</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>34.50000</td>
      <td>7.8292</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>47.00000</td>
      <td>7.0000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>62.00000</td>
      <td>9.6875</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>27.00000</td>
      <td>8.6625</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>22.00000</td>
      <td>12.2875</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>413</th>
      <td>3</td>
      <td>30.27259</td>
      <td>8.0500</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>414</th>
      <td>1</td>
      <td>39.00000</td>
      <td>108.9000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>415</th>
      <td>3</td>
      <td>38.50000</td>
      <td>7.2500</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>416</th>
      <td>3</td>
      <td>30.27259</td>
      <td>8.0500</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>417</th>
      <td>3</td>
      <td>30.27259</td>
      <td>22.3583</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>418 rows × 9 columns</p>
</div>



# 독립변수 정규화 


```python
from sklearn import preprocessing
X_test = preprocessing.StandardScaler().fit(x).transform(x)

```

# 모델링 


```python


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1000.0, random_state=0) # c값을 1000
lr.fit( X, y)
```




    LogisticRegression(C=1000.0, random_state=0)




```python
y_hat = lr.predict( X_test )
print(y_hat)

for  i,a  in  enumerate(y_hat):
    print (str(i+892) + ',' + str(a))

test_id=x_ktest.PassengerId
pred=y_hat 
test_Id
```

    [0 0 0 0 1 0 1 0 1 0 0 0 1 0 1 1 0 0 1 1 0 0 1 1 1 0 1 0 0 0 0 0 0 0 1 0 0
     1 0 0 0 0 0 1 1 0 0 0 1 0 0 0 1 1 0 0 0 0 0 1 0 0 0 1 0 1 1 0 1 1 1 0 1 1
     1 0 0 1 0 1 1 0 0 0 0 0 1 1 1 1 1 0 1 0 1 0 1 0 1 0 1 0 0 0 1 0 0 0 0 0 0
     1 1 1 1 0 0 1 0 1 1 0 1 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 1 0 1 1 0 0 0 0 0 0
     0 0 1 0 0 0 0 0 1 1 0 1 1 1 1 0 0 1 0 0 1 1 0 0 0 0 0 1 1 0 1 1 0 0 1 0 1
     0 1 0 0 0 0 0 0 0 1 0 1 1 0 0 1 0 0 1 0 1 1 0 1 0 0 0 0 1 0 0 1 0 1 0 1 0
     1 0 1 1 0 1 0 0 0 1 0 0 0 0 0 0 1 1 1 1 0 0 0 0 1 0 1 1 1 0 0 0 0 0 0 0 1
     0 0 0 1 1 0 0 0 0 0 0 0 0 1 1 0 1 0 0 0 0 1 0 1 1 1 0 0 0 0 0 0 1 0 0 0 0
     1 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 1 0 1 0 0 0 1 1 0
     1 0 0 0 0 0 0 0 0 0 1 0 1 0 1 0 1 1 0 0 0 1 0 1 0 0 0 0 1 1 0 1 0 0 1 1 0
     0 1 0 0 1 1 1 0 0 0 0 0 0 1 0 1 0 0 0 0 0 1 0 0 0 1 0 1 0 0 1 0 1 0 1 0 0
     0 1 1 1 1 1 0 1 0 0 0]
    892,0
    893,0
    894,0
    895,0
    896,1
    897,0
    898,1
    899,0
    900,1
    901,0
    902,0
    903,0
    904,1
    905,0
    906,1
    907,1
    908,0
    909,0
    910,1
    911,1
    912,0
    913,0
    914,1
    915,1
    916,1
    917,0
    918,1
    919,0
    920,0
    921,0
    922,0
    923,0
    924,0
    925,0
    926,1
    927,0
    928,0
    929,1
    930,0
    931,0
    932,0
    933,0
    934,0
    935,1
    936,1
    937,0
    938,0
    939,0
    940,1
    941,0
    942,0
    943,0
    944,1
    945,1
    946,0
    947,0
    948,0
    949,0
    950,0
    951,1
    952,0
    953,0
    954,0
    955,1
    956,0
    957,1
    958,1
    959,0
    960,1
    961,1
    962,1
    963,0
    964,1
    965,1
    966,1
    967,0
    968,0
    969,1
    970,0
    971,1
    972,1
    973,0
    974,0
    975,0
    976,0
    977,0
    978,1
    979,1
    980,1
    981,1
    982,1
    983,0
    984,1
    985,0
    986,1
    987,0
    988,1
    989,0
    990,1
    991,0
    992,1
    993,0
    994,0
    995,0
    996,1
    997,0
    998,0
    999,0
    1000,0
    1001,0
    1002,0
    1003,1
    1004,1
    1005,1
    1006,1
    1007,0
    1008,0
    1009,1
    1010,0
    1011,1
    1012,1
    1013,0
    1014,1
    1015,0
    1016,0
    1017,1
    1018,0
    1019,1
    1020,0
    1021,0
    1022,0
    1023,0
    1024,0
    1025,0
    1026,0
    1027,0
    1028,0
    1029,0
    1030,1
    1031,0
    1032,1
    1033,1
    1034,0
    1035,0
    1036,0
    1037,0
    1038,0
    1039,0
    1040,0
    1041,0
    1042,1
    1043,0
    1044,0
    1045,0
    1046,0
    1047,0
    1048,1
    1049,1
    1050,0
    1051,1
    1052,1
    1053,1
    1054,1
    1055,0
    1056,0
    1057,1
    1058,0
    1059,0
    1060,1
    1061,1
    1062,0
    1063,0
    1064,0
    1065,0
    1066,0
    1067,1
    1068,1
    1069,0
    1070,1
    1071,1
    1072,0
    1073,0
    1074,1
    1075,0
    1076,1
    1077,0
    1078,1
    1079,0
    1080,0
    1081,0
    1082,0
    1083,0
    1084,0
    1085,0
    1086,1
    1087,0
    1088,1
    1089,1
    1090,0
    1091,0
    1092,1
    1093,0
    1094,0
    1095,1
    1096,0
    1097,1
    1098,1
    1099,0
    1100,1
    1101,0
    1102,0
    1103,0
    1104,0
    1105,1
    1106,0
    1107,0
    1108,1
    1109,0
    1110,1
    1111,0
    1112,1
    1113,0
    1114,1
    1115,0
    1116,1
    1117,1
    1118,0
    1119,1
    1120,0
    1121,0
    1122,0
    1123,1
    1124,0
    1125,0
    1126,0
    1127,0
    1128,0
    1129,0
    1130,1
    1131,1
    1132,1
    1133,1
    1134,0
    1135,0
    1136,0
    1137,0
    1138,1
    1139,0
    1140,1
    1141,1
    1142,1
    1143,0
    1144,0
    1145,0
    1146,0
    1147,0
    1148,0
    1149,0
    1150,1
    1151,0
    1152,0
    1153,0
    1154,1
    1155,1
    1156,0
    1157,0
    1158,0
    1159,0
    1160,0
    1161,0
    1162,0
    1163,0
    1164,1
    1165,1
    1166,0
    1167,1
    1168,0
    1169,0
    1170,0
    1171,0
    1172,1
    1173,0
    1174,1
    1175,1
    1176,1
    1177,0
    1178,0
    1179,0
    1180,0
    1181,0
    1182,0
    1183,1
    1184,0
    1185,0
    1186,0
    1187,0
    1188,1
    1189,0
    1190,0
    1191,0
    1192,0
    1193,0
    1194,0
    1195,0
    1196,1
    1197,1
    1198,0
    1199,0
    1200,0
    1201,0
    1202,0
    1203,0
    1204,0
    1205,1
    1206,1
    1207,1
    1208,0
    1209,0
    1210,0
    1211,0
    1212,0
    1213,0
    1214,0
    1215,0
    1216,1
    1217,0
    1218,1
    1219,0
    1220,0
    1221,0
    1222,1
    1223,1
    1224,0
    1225,1
    1226,0
    1227,0
    1228,0
    1229,0
    1230,0
    1231,0
    1232,0
    1233,0
    1234,0
    1235,1
    1236,0
    1237,1
    1238,0
    1239,1
    1240,0
    1241,1
    1242,1
    1243,0
    1244,0
    1245,0
    1246,1
    1247,0
    1248,1
    1249,0
    1250,0
    1251,0
    1252,0
    1253,1
    1254,1
    1255,0
    1256,1
    1257,0
    1258,0
    1259,1
    1260,1
    1261,0
    1262,0
    1263,1
    1264,0
    1265,0
    1266,1
    1267,1
    1268,1
    1269,0
    1270,0
    1271,0
    1272,0
    1273,0
    1274,0
    1275,1
    1276,0
    1277,1
    1278,0
    1279,0
    1280,0
    1281,0
    1282,0
    1283,1
    1284,0
    1285,0
    1286,0
    1287,1
    1288,0
    1289,1
    1290,0
    1291,0
    1292,1
    1293,0
    1294,1
    1295,0
    1296,1
    1297,0
    1298,0
    1299,0
    1300,1
    1301,1
    1302,1
    1303,1
    1304,1
    1305,0
    1306,1
    1307,0
    1308,0
    1309,0
    


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-133-92f1f3a4dac4> in <module>
          7 test_id=x_ktest.PassengerId
          8 pred=y_hat
    ----> 9 test_Id
    

    NameError: name 'test_Id' is not defined



```python
submission=pd.DataFrame({'Passengerid': test_id, 'Survived': pred})

submission.to_csv('c:\\data\\submission3.csv', index=False)
submission.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Passengerid</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


