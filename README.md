Porcentaje de valores nulos:
wild_card_winner         72.198276
division_winner          49.353448
sacrifice_flies          49.209770
division_id              48.347701
batters_hit_by_pitch     38.290230
caught_stealing          25.431034
world_series_winner       8.908046
home_games                8.189655
home_attendance           3.879310
stolen_bases              2.729885
league_winner             1.005747
strikeouts_by_batters     0.574713
dtype: float64

Estadísticas descriptivas (numéricas):
          rownames         year         rank  games_played   home_games         wins  ...  walks_allowed  strikeouts_by_pitchers       errors  double_plays  fielding_percentage  home_attendance
count  2784.000000  2784.000000  2784.000000   2784.000000  2556.000000  2784.000000  ...    2784.000000             2784.000000  2784.000000   2784.000000          2784.000000     2.676000e+03
mean   1392.500000  1962.717672     3.980963    153.118534    78.014476    76.189296  ...     487.762931              781.191092   168.217672    136.936063             0.971032     1.375102e+06
std     803.815899    39.800654     2.237914     18.201875     6.971355    15.909342  ...     112.202251              300.560361    90.254000     31.340128             0.019017     9.662279e+05
min       1.000000  1876.000000     1.000000     57.000000    24.000000     9.000000  ...      24.000000               22.000000    20.000000     18.000000             0.825000     0.000000e+00
25%     696.750000  1930.000000     2.000000    154.000000    77.000000    67.000000  ...     439.000000              531.000000   111.000000    122.000000             0.969000     5.348265e+05
50%    1392.500000  1970.000000     4.000000    161.000000    81.000000    78.000000  ...     500.000000              784.000000   139.000000    142.000000             0.977000     1.184548e+06
75%    2088.250000  1997.000000     5.000000    162.000000    81.000000    88.000000  ...     558.000000             1001.000000   189.000000    158.000000             0.981000     2.068023e+06
max    2784.000000  2020.000000    12.000000    165.000000    84.000000   116.000000  ...     827.000000             1687.000000   639.000000    217.000000             0.991000     4.483350e+06

[8 rows x 34 columns]

Estadísticas descriptivas (categóricas):
       league_id division_id division_winner wild_card_winner league_winner world_series_winner        team_name      ball_park
count       2784        1438            1410              774          2756                2536             2784           2784
unique         2           3               2                2             2                   2               86            183
top           NL           E               N                N             N                   N  Cincinnati Reds  Wrigley Field
freq        1504         588            1150              698          2493                2416              130            105

Dataset limpio y preprocesado:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 2784 entries, 0 to 2783
Data columns (total 39 columns):
 #   Column                  Non-Null Count  Dtype  
---  ------                  --------------  -----  
 0   year                    2784 non-null   float64
 1   league_id               2784 non-null   int64  
 2   division_id             2784 non-null   int64  
 3   rank                    2784 non-null   float64
 4   games_played            2784 non-null   float64
 5   home_games              2784 non-null   float64
 6   wins                    2784 non-null   float64
 7   losses                  2784 non-null   float64
 8   division_winner         2784 non-null   int64  
 9   wild_card_winner        2784 non-null   int64  
 10  league_winner           2784 non-null   int64  
 11  world_series_winner     2784 non-null   int64  
 12  runs_scored             2784 non-null   float64
 13  at_bats                 2784 non-null   float64
 14  hits                    2784 non-null   float64
 15  doubles                 2784 non-null   float64
 16  triples                 2784 non-null   float64
 17  homeruns                2784 non-null   float64
 18  walks                   2784 non-null   float64
 19  strikeouts_by_batters   2784 non-null   float64
 20  stolen_bases            2784 non-null   float64
 21  caught_stealing         2784 non-null   float64
 22  batters_hit_by_pitch    2784 non-null   float64
 23  sacrifice_flies         2784 non-null   float64
 24  opponents_runs_scored   2784 non-null   float64
 25  earned_runs_allowed     2784 non-null   float64
 26  earned_run_average      2784 non-null   float64
 27  complete_games          2784 non-null   float64
 28  shutouts                2784 non-null   float64
 29  saves                   2784 non-null   float64
 30  outs_pitches            2784 non-null   float64
 31  hits_allowed            2784 non-null   float64
 32  homeruns_allowed        2784 non-null   float64
 33  walks_allowed           2784 non-null   float64
 34  strikeouts_by_pitchers  2784 non-null   float64
 35  errors                  2784 non-null   float64
 36  double_plays            2784 non-null   float64
 37  fielding_percentage     2784 non-null   float64
 38  home_attendance         2784 non-null   float64
dtypes: float64(33), int64(6)
memory usage: 848.4 KB
None

Ejemplo de primeras filas:
      year  league_id  division_id  rank  games_played  home_games  wins  ...  homeruns_allowed  walks_allowed  strikeouts_by_pitchers  errors  double_plays  fielding_percentage  home_attendance
0  1876.0          1            1   4.0          70.0   78.014476  39.0  ...               7.0          104.0                    77.0   442.0          42.0                0.860     1.375102e+06
1  1876.0          1            1   1.0          66.0   78.014476  52.0  ...               6.0           29.0                    51.0   282.0          33.0                0.899     1.375102e+06
2  1876.0          1            1   8.0          65.0   78.014476   9.0  ...               9.0           34.0                    60.0   469.0          45.0                0.841     1.375102e+06
3  1876.0          1            1   2.0          69.0   78.014476  47.0  ...               2.0           27.0                   114.0   337.0          27.0                0.888     1.375102e+06
4  1876.0          1            1   5.0          69.0   78.014476  30.0  ...               3.0           38.0                   125.0   397.0          44.0                0.875     1.375102e+06

[5 rows x 39 columns]

Trabajaremos con tres targets: division_winner, league_winner, world_series_winner


🔍 Selección univariante para target: 'division_winner'

🔍 Selección univariante para target: 'league_winner'

🔍 Selección univariante para target: 'world_series_winner'

📊 Resultados para target: 'division_winner'

🔹 Modelo: KNN (k=1)
   Accuracy: 0.8696
   F1: 0.3040
   Precision: 0.3012
   Recall: 0.3077

🔹 Modelo: KNN (k=3)
   Accuracy: 0.8948
   F1: 0.3255
   Precision: 0.4041
   Recall: 0.2731

🔹 Modelo: Naive Bayes
   Accuracy: 0.7022
   F1: 0.3537
   Precision: 0.2222
   Recall: 0.8692

🔹 Modelo: Decision Tree (entropy)
   Accuracy: 0.9968
   F1: 0.9827
   Precision: 0.9851
   Recall: 0.9808

🔹 Modelo: Decision Tree (J48-like)
   Accuracy: 0.9946
   F1: 0.9720
   Precision: 0.9530
   Recall: 0.9923

🔹 Modelo: Logistic Regression
   Accuracy: 0.9231
   F1: 0.4729
   Precision: 0.6600
   Recall: 0.3692

📊 Resultados para target: 'league_winner'

🔹 Modelo: KNN (k=1)
   Accuracy: 0.8402
   F1: 0.1073
   Precision: 0.1142
   Recall: 0.1025

🔹 Modelo: KNN (k=3)
   Accuracy: 0.8865
   F1: 0.0635
   Precision: 0.1406
   Recall: 0.0417

🔹 Modelo: Naive Bayes
   Accuracy: 0.8894
   F1: 0.3811
   Precision: 0.4159
   Recall: 0.3573

🔹 Modelo: Decision Tree (entropy)
   Accuracy: 0.9501
   F1: 0.7341
   Precision: 0.7386
   Recall: 0.7302

🔹 Modelo: Decision Tree (J48-like)
   Accuracy: 0.9228
   F1: 0.7048
   Precision: 0.5529
   Recall: 0.9734

🔹 Modelo: Logistic Regression
   Accuracy: 0.9152
   F1: 0.3299
   Precision: 0.6352
   Recall: 0.2282

📊 Resultados para target: 'world_series_winner'

🔹 Modelo: KNN (k=1)
   Accuracy: 0.9213
   F1: 0.0512
   Precision: 0.0527
   Recall: 0.0500

🔹 Modelo: KNN (k=3)
   Accuracy: 0.9508
   F1: 0.0281
   Precision: 0.1000
   Recall: 0.0167

🔹 Modelo: Naive Bayes
   Accuracy: 0.9055
   F1: 0.3070
   Precision: 0.2426
   Recall: 0.4583

🔹 Modelo: Decision Tree (entropy)
   Accuracy: 0.9468
   F1: 0.3941
   Precision: 0.4095
   Recall: 0.4000

🔹 Modelo: Decision Tree (J48-like)
   Accuracy: 0.8962
   F1: 0.4071
   Precision: 0.2703
   Recall: 0.8250

🔹 Modelo: Logistic Regression
   Accuracy: 0.9551
   F1: 0.0837
   Precision: 0.4417
   Recall: 0.0500

🌱 Evaluando J48 con SMOTE para target: division_winner

📊 Resultados para target: 'division_winner'

🔹 Modelo: Decision Tree (J48-SMOTE)
   Accuracy: 0.9910
   F1: 0.9544
   Precision: 0.9167
   Recall: 0.9962

🌱 Evaluando J48 con SMOTE para target: league_winner

📊 Resultados para target: 'league_winner'

🔹 Modelo: Decision Tree (J48-SMOTE)
   Accuracy: 0.9318
   F1: 0.7231
   Precision: 0.5869
   Recall: 0.9429

🌱 Evaluando J48 con SMOTE para target: world_series_winner

📊 Resultados para target: 'world_series_winner'

🔹 Modelo: Decision Tree (J48-SMOTE)
   Accuracy: 0.9102
   F1: 0.4215
   Precision: 0.2975
   Recall: 0.7417

🌲 Random Forest para target: division_winner

📊 Resultados para target: 'division_winner'

🔹 Modelo: Random Forest (100 trees)
   Accuracy: 0.9968
   F1: 0.9823
   Precision: 1.0000
   Recall: 0.9654

🌲 Random Forest para target: league_winner

📊 Resultados para target: 'league_winner'

🔹 Modelo: Random Forest (100 trees)
   Accuracy: 0.9626
   F1: 0.7749
   Precision: 0.8996
   Recall: 0.6845

🌲 Random Forest para target: world_series_winner

📊 Resultados para target: 'world_series_winner'

🔹 Modelo: Random Forest (100 trees)
   Accuracy: 0.9558
   F1: 0.1927
   Precision: 0.5076
   Recall: 0.1250

🤝 Clasificadores combinados para target: division_winner

📊 Resultados para target: 'division_winner'

🔹 Modelo: VotingClassifier
   Accuracy: 0.9415
   F1: 0.5831
   Precision: 0.8732
   Recall: 0.4385

🔹 Modelo: StackingClassifier
   Accuracy: 0.9953
   F1: 0.9744
   Precision: 0.9851
   Recall: 0.9654

🤝 Clasificadores combinados para target: league_winner

📊 Resultados para target: 'league_winner'

🔹 Modelo: VotingClassifier
   Accuracy: 0.9210
   F1: 0.3078
   Precision: 0.8893
   Recall: 0.1901

🔹 Modelo: StackingClassifier
   Accuracy: 0.9472
   F1: 0.6675
   Precision: 0.8135
   Recall: 0.5902

🤝 Clasificadores combinados para target: world_series_winner

📊 Resultados para target: 'world_series_winner'

🔹 Modelo: VotingClassifier
   Accuracy: 0.9569
   F1: 0.0436
   Precision: 0.2800
   Recall: 0.0250

🔹 Modelo: StackingClassifier
   Accuracy: 0.9558
   F1: 0.1023
   Precision: 0.4309
   Recall: 0.0667


📊 Tabla resumen de todos los modelos:
           Target                 Tipo                    Modelo  Accuracy     F1  Precision  Recall
     leaguewinner Modelos Supervisados                 KNN (k=1)    0.8402 0.1073     0.1142  0.1025
     leaguewinner Modelos Supervisados                 KNN (k=3)    0.8865 0.0635     0.1406  0.0417
     leaguewinner Modelos Supervisados               Naive Bayes    0.8894 0.3811     0.4159  0.3573
     leaguewinner Modelos Supervisados   Decision Tree (entropy)    0.9501 0.7341     0.7386  0.7302
     leaguewinner Modelos Supervisados  Decision Tree (J48-like)    0.9228 0.7048     0.5529  0.9734
     leaguewinner Modelos Supervisados       Logistic Regression    0.9152 0.3299     0.6352  0.2282
   divisionwinner Modelos Supervisados                 KNN (k=1)    0.8696 0.3040     0.3012  0.3077
   divisionwinner Modelos Supervisados                 KNN (k=3)    0.8948 0.3255     0.4041  0.2731
   divisionwinner Modelos Supervisados               Naive Bayes    0.7022 0.3537     0.2222  0.8692
   divisionwinner Modelos Supervisados   Decision Tree (entropy)    0.9968 0.9827     0.9851  0.9808
   divisionwinner Modelos Supervisados  Decision Tree (J48-like)    0.9946 0.9720     0.9530  0.9923
   divisionwinner Modelos Supervisados       Logistic Regression    0.9231 0.4729     0.6600  0.3692
worldserieswinner Modelos Supervisados                 KNN (k=1)    0.9213 0.0512     0.0527  0.0500
worldserieswinner Modelos Supervisados                 KNN (k=3)    0.9508 0.0281     0.1000  0.0167
worldserieswinner Modelos Supervisados               Naive Bayes    0.9055 0.3070     0.2426  0.4583
worldserieswinner Modelos Supervisados   Decision Tree (entropy)    0.9468 0.3941     0.4095  0.4000
worldserieswinner Modelos Supervisados  Decision Tree (J48-like)    0.8962 0.4071     0.2703  0.8250
worldserieswinner Modelos Supervisados       Logistic Regression    0.9551 0.0837     0.4417  0.0500
     leaguewinner          J48 + SMOTE Decision Tree (J48-SMOTE)    0.9318 0.7231     0.5869  0.9429
worldserieswinner          J48 + SMOTE Decision Tree (J48-SMOTE)    0.9102 0.4215     0.2975  0.7417
   divisionwinner          J48 + SMOTE Decision Tree (J48-SMOTE)    0.9910 0.9544     0.9167  0.9962
worldserieswinner        Random Forest Random Forest (100 trees)    0.9558 0.1927     0.5076  0.1250
   divisionwinner        Random Forest Random Forest (100 trees)    0.9968 0.9823     1.0000  0.9654
     leaguewinner        Random Forest Random Forest (100 trees)    0.9626 0.7749     0.8996  0.6845
worldserieswinner           Combinados          VotingClassifier    0.9569 0.0436     0.2800  0.0250
worldserieswinner           Combinados        StackingClassifier    0.9558 0.1023     0.4309  0.0667
     leaguewinner           Combinados          VotingClassifier    0.9210 0.3078     0.8893  0.1901
     leaguewinner           Combinados        StackingClassifier    0.9472 0.6675     0.8135  0.5902
   divisionwinner           Combinados          VotingClassifier    0.9415 0.5831     0.8732  0.4385
   divisionwinner           Combinados        StackingClassifier    0.9953 0.9744     0.9851  0.9654