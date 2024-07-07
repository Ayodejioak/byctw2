{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "39d9f748-e625-419d-a803-cdb21cb2f1b3",
   "metadata": {},
   "source": [
    "### Step 0: Import necessary libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "92cf7aee-5d91-4627-aedd-cef84c73ac6c",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import numpy as np\n",
    "from scipy import stats\n",
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import mean_squared_error, r2_score"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "07a8fe68-02ab-4be6-b0cc-5591fc06c552",
   "metadata": {},
   "source": [
    "#### Set 1: Data Cleaning and Exploration\n",
    "#####      Step 1: Load the data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "1f92d313-5281-4c6e-9694-a0cb2b980b14",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   Rank                           Name                      basename  \\\n",
      "0     1                     Wii Sports                    wii-sports   \n",
      "1     2              Super Mario Bros.              super-mario-bros   \n",
      "2     3                 Mario Kart Wii                mario-kart-wii   \n",
      "3     4  PlayerUnknown's Battlegrounds  playerunknowns-battlegrounds   \n",
      "4     5              Wii Sports Resort             wii-sports-resort   \n",
      "\n",
      "      Genre ESRB_Rating Platform         Publisher         Developer  \\\n",
      "0    Sports           E      Wii          Nintendo      Nintendo EAD   \n",
      "1  Platform         NaN      NES          Nintendo      Nintendo EAD   \n",
      "2    Racing           E      Wii          Nintendo      Nintendo EAD   \n",
      "3   Shooter         NaN       PC  PUBG Corporation  PUBG Corporation   \n",
      "4    Sports           E      Wii          Nintendo      Nintendo EAD   \n",
      "\n",
      "   VGChartz_Score  Critic_Score  ...  NA_Sales  PAL_Sales  JP_Sales  \\\n",
      "0             NaN           7.7  ...       NaN        NaN       NaN   \n",
      "1             NaN          10.0  ...       NaN        NaN       NaN   \n",
      "2             NaN           8.2  ...       NaN        NaN       NaN   \n",
      "3             NaN           NaN  ...       NaN        NaN       NaN   \n",
      "4             NaN           8.0  ...       NaN        NaN       NaN   \n",
      "\n",
      "   Other_Sales    Year  Last_Update  \\\n",
      "0          NaN  2006.0          NaN   \n",
      "1          NaN  1985.0          NaN   \n",
      "2          NaN  2008.0  11th Apr 18   \n",
      "3          NaN  2017.0  13th Nov 18   \n",
      "4          NaN  2009.0          NaN   \n",
      "\n",
      "                                                 url  status Vgchartzscore  \\\n",
      "0  http://www.vgchartz.com/game/2667/wii-sports/?...       1           NaN   \n",
      "1  http://www.vgchartz.com/game/6455/super-mario-...       1           NaN   \n",
      "2  http://www.vgchartz.com/game/6968/mario-kart-w...       1           8.7   \n",
      "3  http://www.vgchartz.com/game/215988/playerunkn...       1           NaN   \n",
      "4  http://www.vgchartz.com/game/24656/wii-sports-...       1           8.8   \n",
      "\n",
      "                                         img_url  \n",
      "0  /games/boxart/full_2258645AmericaFrontccc.jpg  \n",
      "1                   /games/boxart/8972270ccc.jpg  \n",
      "2  /games/boxart/full_8932480AmericaFrontccc.jpg  \n",
      "3  /games/boxart/full_8052843AmericaFrontccc.jpg  \n",
      "4  /games/boxart/full_7295041AmericaFrontccc.jpg  \n",
      "\n",
      "[5 rows x 23 columns]\n"
     ]
    }
   ],
   "source": [
    "df = pd.read_csv('VideoGameSales.csv') # Load the video game sales dataset from a CSV file\n",
    "print(df.head())  # Display the first few rows of the dataset\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1fa41232-d10e-47c4-8a9c-aa4c78b0700b",
   "metadata": {},
   "source": [
    "##### Step 2: Data summary"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "64c4e4b7-8107-42a0-b58e-7588bb51d64f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 55792 entries, 0 to 55791\n",
      "Data columns (total 23 columns):\n",
      " #   Column          Non-Null Count  Dtype  \n",
      "---  ------          --------------  -----  \n",
      " 0   Rank            55792 non-null  int64  \n",
      " 1   Name            55792 non-null  object \n",
      " 2   basename        55792 non-null  object \n",
      " 3   Genre           55792 non-null  object \n",
      " 4   ESRB_Rating     23623 non-null  object \n",
      " 5   Platform        55792 non-null  object \n",
      " 6   Publisher       55792 non-null  object \n",
      " 7   Developer       55775 non-null  object \n",
      " 8   VGChartz_Score  0 non-null      float64\n",
      " 9   Critic_Score    6536 non-null   float64\n",
      " 10  User_Score      335 non-null    float64\n",
      " 11  Total_Shipped   1827 non-null   float64\n",
      " 12  Global_Sales    19415 non-null  float64\n",
      " 13  NA_Sales        12964 non-null  float64\n",
      " 14  PAL_Sales       13189 non-null  float64\n",
      " 15  JP_Sales        7043 non-null   float64\n",
      " 16  Other_Sales     15522 non-null  float64\n",
      " 17  Year            54813 non-null  float64\n",
      " 18  Last_Update     9186 non-null   object \n",
      " 19  url             55792 non-null  object \n",
      " 20  status          55792 non-null  int64  \n",
      " 21  Vgchartzscore   799 non-null    float64\n",
      " 22  img_url         55792 non-null  object \n",
      "dtypes: float64(11), int64(2), object(10)\n",
      "memory usage: 9.8+ MB\n",
      "None\n",
      "               Rank  VGChartz_Score  Critic_Score  User_Score  Total_Shipped  \\\n",
      "count  55792.000000             0.0   6536.000000  335.000000    1827.000000   \n",
      "mean   27896.500000             NaN      7.213709    8.253433       1.887258   \n",
      "std    16105.907446             NaN      1.454079    1.401489       4.195693   \n",
      "min        1.000000             NaN      1.000000    2.000000       0.030000   \n",
      "25%    13948.750000             NaN      6.400000    7.800000       0.200000   \n",
      "50%    27896.500000             NaN      7.500000    8.500000       0.590000   \n",
      "75%    41844.250000             NaN      8.300000    9.100000       1.800000   \n",
      "max    55792.000000             NaN     10.000000   10.000000      82.860000   \n",
      "\n",
      "       Global_Sales      NA_Sales     PAL_Sales     JP_Sales   Other_Sales  \\\n",
      "count  19415.000000  12964.000000  13189.000000  7043.000000  15522.000000   \n",
      "mean       0.365503      0.275541      0.155263     0.110402      0.044719   \n",
      "std        0.833022      0.512809      0.399257     0.184673      0.129554   \n",
      "min        0.000000      0.000000      0.000000     0.000000      0.000000   \n",
      "25%        0.030000      0.050000      0.010000     0.020000      0.000000   \n",
      "50%        0.120000      0.120000      0.040000     0.050000      0.010000   \n",
      "75%        0.360000      0.290000      0.140000     0.120000      0.040000   \n",
      "max       20.320000      9.760000      9.850000     2.690000      3.120000   \n",
      "\n",
      "               Year   status  Vgchartzscore  \n",
      "count  54813.000000  55792.0     799.000000  \n",
      "mean    2005.659095      1.0       7.425907  \n",
      "std        8.355585      0.0       1.384226  \n",
      "min     1970.000000      1.0       2.600000  \n",
      "25%     2000.000000      1.0       6.800000  \n",
      "50%     2008.000000      1.0       7.800000  \n",
      "75%     2011.000000      1.0       8.500000  \n",
      "max     2020.000000      1.0       9.600000  \n"
     ]
    }
   ],
   "source": [
    "print(df.info())  # Display information about the dataset, including column names and non-null counts\n",
    "print(df.describe())  # Generate descriptive statistics of the numerical columns\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "a9975226-335c-4777-a940-71a696257671",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Handle missing values\n",
    "df_cleaned = df.dropna(subset=['Global_Sales', 'Critic_Score', 'User_Score'])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bca309bb-f140-43d3-b983-7dd149287eab",
   "metadata": {},
   "source": [
    "##### Step 3: Visualize data distribution"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "f7b2b37e-b64f-4401-9952-a225bf4efba1",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Z Y B S Medical\\anaconda3\\Lib\\site-packages\\seaborn\\_oldcore.py:1119: FutureWarning:\n",
      "\n",
      "use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.\n",
      "\n",
      "C:\\Users\\Z Y B S Medical\\anaconda3\\Lib\\site-packages\\seaborn\\_oldcore.py:1119: FutureWarning:\n",
      "\n",
      "use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.\n",
      "\n",
      "C:\\Users\\Z Y B S Medical\\anaconda3\\Lib\\site-packages\\seaborn\\_oldcore.py:1119: FutureWarning:\n",
      "\n",
      "use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.\n",
      "\n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAABdAAAAHqCAYAAAAEZWxJAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy81sbWrAAAACXBIWXMAAA9hAAAPYQGoP6dpAADdRklEQVR4nOzdeXhU9fn+8Xtmkkz2fSckhEDYwy6LCyCCsrmgVesGrbXuVq3Vqq2iv1aqVqutVlur1n4Vl1qxLlVAEURZZF/DviRAFrLv28z5/REyGkggCUnOJPN+XVeuy8yczLkzIM/MM5/zfCyGYRgCAAAAAAAAAACNWM0OAAAAAAAAAACAO6KBDgAAAAAAAABAE2igAwAAAAAAAADQBBroAAAAAAAAAAA0gQY6AAAAAAAAAABNoIEOAAAAAAAAAEATaKADAAAAAAAAANAEGugAAAAAAAAAADSBBjoAAAAAAAAAAE2ggQ5T/fOf/5TFYnF9+fr6KjY2VpMmTdL8+fOVm5t70s/MmzdPFoulVeepqKjQvHnztGzZslb9XFPn6tWrl2bOnNmqxzmdBQsW6LnnnmvyPovFonnz5rXr+drbl19+qVGjRikgIEAWi0UffvjhKY/PycnRr3/9aw0ZMkSBgYHy9fVV37599Ytf/EJ79uxp0Tnnzp2rXr16NbrtiSeeaPLcy5Ytk8ViafWff3MyMzN12223KTU1VX5+fgoPD9eQIUN00003KTMzs13OAQDujPpdz1Pqd0Mdff/995u8/4477mj1n21Hqq2t1d/+9jeNHj1a4eHh8vf3V1JSki655BItXLjQ7HgA0Cmo1fU8pVY3yMnJ0UMPPaRhw4YpODhYPj4+SkhI0OzZs/XRRx/J4XC4jj2T98kNf355eXmt/tnTPWZLLFq0SFOnTlV8fLzsdrvi4+M1ceJE/eEPf2jTuZvqLwA/5GV2AECSXn/9dfXv31+1tbXKzc3VN998oyeffFJ//OMf9e677+qCCy5wHfuzn/1MF110Uasev6KiQo899pgkaeLEiS3+ubacqy0WLFigbdu26e677z7pvlWrVikhIaHDM7SVYRi68sorlZqaqo8++kgBAQHq169fs8d/9913mjlzpgzD0B133KFx48bJx8dHu3bt0ptvvqmzzjpLhYWFpz3vb3/7W/3iF79odNsTTzyhK664Qpdeemmj20eMGKFVq1Zp4MCBbfodf+jw4cMaMWKEQkND9ctf/lL9+vVTcXGxduzYoffee0/79+9Xz549z/g8ANAVUL89p353Jddff70++OAD3X333Xrsscdkt9u1f/9+ff7551q0aJEuu+wysyMCQKehVntOrV69erUuvvhiGYahW2+9VWPHjlVgYKAyMjL08ccfa/bs2frb3/6mG2+8sRN/i/b38ssv69Zbb9Xll1+uF154QeHh4crMzNTKlSv1/vvv69e//rXZEdEN0UCHWxg8eLBGjRrl+v7yyy/XPffco3POOUezZ8/Wnj17FBMTI0lKSEjo8CJXUVEhf3//TjnX6YwdO9bU85/O0aNHVVBQoMsuu0yTJ08+5bElJSW65JJL5Ovrq5UrVzZ6bidOnKibb7652dVtDRr+bFJSUlqcMTg4uN2ex1deeUV5eXn67rvvlJyc7Lr90ksv1UMPPSSn09ku52mJyspK+fr6utWqPwCehfrdvO5Uv92NYRiqqqqSn5/fSfcdOHBA7777rh555BFXQ0eSJk+erJtuuqlT6/SpcgJAZ6FWN6871eqioiJdeumlCgwM1Lfffqu4uLhG91933XXasmWL8vPzOzJyp5g/f77OO++8k3oH119/fafWeXgWRrjAbSUmJuqZZ55RaWmp/va3v7lub+qynqVLl2rixImKiIiQn5+fEhMTdfnll6uiokIHDx5UVFSUJOmxxx5zXcI2d+7cRo+3YcMGXXHFFQoLC3M1Z091CdHChQuVlpYmX19f9e7dW3/+858b3d9wydzBgwcb3X7iZVITJ07Up59+qkOHDjW6xK5BU5eVbdu2TZdcconCwsLk6+urYcOG6Y033mjyPG+//bYefvhhxcfHKzg4WBdccIF27drV/BP/A998840mT56soKAg+fv7a/z48fr0009d98+bN8/1oueBBx6QxWI55WVPr7zyirKzs/XUU081+2LpiiuucP333LlzFRgYqK1bt2rq1KkKCgpyvXA48RIri8Wi8vJyvfHGG67nsGEFRHOXpq1Zs0azZs1SRESEfH19lZKS0uTKhB/Kz8+X1WpVdHR0k/dbrY3/WW3JOU73PEvf/31avHixfvrTnyoqKkr+/v6qrq6WJL377rsaN26cAgICFBgYqAsvvFAbN25s9Bj79+/X1Vdf7brMLSYmRpMnT9amTZtO+TsDQGtQv+t1p/rdFv/+9781ZswYhYSEyN/fX71799ZPf/rTRseUlJTovvvuU3Jysnx8fNSjRw/dfffdKi8vb3ScxWLRHXfcoZdfflkDBgyQ3W4/6Xlr0NAYOLFx0ODEOl1UVKRf/vKX6t27t+x2u6KjozV9+nTt3LnTdUxBQYFuu+029ejRQz4+Purdu7cefvhhVw1uSc49e/bommuuUXR0tOx2uwYMGKAXX3yx0c87nU797ne/U79+/eTn56fQ0FClpaXp+eefb+5pBoA2oVbX6061+pVXXlFOTo6eeuqpZmtgWlqaJk2adNpsH330kcaNGyd/f38FBQVpypQpWrVqVZPHZmZmavbs2QoODlZISIiuu+46HTt2rNEx7777rqZOnaq4uDj5+flpwIAB+vWvf31SvW+p/Pz8Ftf5F198Ueedd56io6MVEBCgIUOG6KmnnlJtbe1pz2MYhv76179q2LBh8vPzU1hYmK644grt37+/0XEbN27UzJkzXTU+Pj5eM2bM0OHDh9v0+8E9sQIdbm369Omy2Wz6+uuvmz3m4MGDmjFjhs4991y99tprCg0N1ZEjR/T555+rpqZGcXFx+vzzz3XRRRfpxhtv1M9+9jNJchX6BrNnz9bVV1+tW2655bT/kG/atEl333235s2bp9jYWL311lv6xS9+oZqaGt13332t+h3/+te/6uc//7n27dvXormcu3bt0vjx4xUdHa0///nPioiI0Jtvvqm5c+cqJydH999/f6PjH3roIZ199tn6xz/+oZKSEj3wwAOaNWuW0tPTZbPZmj3P8uXLNWXKFKWlpenVV1+V3W7XX//6V82aNUtvv/22rrrqKv3sZz/T0KFDNXv2bN1555265pprZLfbm33MxYsXy2azadasWS1+fmpqanTxxRfr5ptv1q9//WvV1dU1edyqVat0/vnna9KkSfrtb38rqX7leXMWLVqkWbNmacCAAXr22WeVmJiogwcPavHixafMM27cOL344ouaPXu27r33Xo0bN67Z87TkHC15nn/opz/9qWbMmKH/+7//U3l5uby9vfXEE0/oN7/5jX7yk5/oN7/5jWpqavT000/r3HPP1XfffecaXTN9+nQ5HA499dRTSkxMVF5enlauXKmioqJT/s4A0FrU75N15frdWqtWrdJVV12lq666SvPmzZOvr68OHTqkpUuXuo6pqKjQhAkTdPjwYT300ENKS0vT9u3b9cgjj2jr1q364osvGjU5PvzwQ61YsUKPPPKIYmNjm/0ge8CAAQoNDdVjjz0mq9WqqVOnNttwKC0t1TnnnKODBw/qgQce0JgxY1RWVqavv/5aWVlZ6t+/v6qqqjRp0iTt27dPjz32mNLS0rRixQrNnz9fmzZtOukD76Zy7tixQ+PHj3c1rGJjY7Vo0SLdddddysvL06OPPipJeuqppzRv3jz95je/0Xnnnafa2lrt3LmTOg2gQ1CrT9aVa/WSJUtks9k0ffr0lj9BTViwYIGuvfZaTZ06VW+//baqq6v11FNPaeLEifryyy91zjnnNDr+sssu05VXXqlbbrlF27dv129/+1vt2LFDa9askbe3t6T6D5GnT5+uu+++WwEBAdq5c6eefPJJfffdd41eG7TUuHHj9J///Efz5s3TZZddpsGDBzf7fO/bt0/XXHON68P6zZs36/e//7127typ11577ZTnufnmm/XPf/5Td911l5588kkVFBTo8ccf1/jx47V582bFxMSovLxcU6ZMUXJysl588UXFxMQoOztbX331lUpLS1v9u8GNGYCJXn/9dUOSsXbt2maPiYmJMQYMGOD6/tFHHzV++Ff3/fffNyQZmzZtavYxjh07ZkgyHn300ZPua3i8Rx55pNn7figpKcmwWCwnnW/KlClGcHCwUV5e3uh3O3DgQKPjvvrqK0OS8dVXX7lumzFjhpGUlNRk9hNzX3311YbdbjcyMjIaHTdt2jTD39/fKCoqanSe6dOnNzruvffeMyQZq1atavJ8DcaOHWtER0cbpaWlrtvq6uqMwYMHGwkJCYbT6TQMwzAOHDhgSDKefvrpUz6eYRhG//79jdjY2NMe12DOnDmGJOO1115r8r4Tn7OAgABjzpw5Jx3b1HOekpJipKSkGJWVlS3OYxiG4XQ6jZtvvtmwWq2GJMNisRgDBgww7rnnnpP+rFtyjpY+zw1/n2644YZGP5+RkWF4eXkZd955Z6PbS0tLjdjYWOPKK680DMMw8vLyDEnGc88916rfFwCaQv2u5yn1uyHTv//97ybvv/322xs933/84x8NSa7fqSnz5883rFbrSX+HGv5e/O9//3PdJskICQkxCgoKTpvVMAzj008/NSIjIw1JhiQjIiLC+NGPfmR89NFHjY57/PHHDUnGkiVLmn2sl19+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/nxNmTJFkyZN0n333ScfHx/99a9/1bZt2/T222+3aSVYSEiI/vvf/2rmzJkaPny4a0aZj4+P9uzZozfffFObN2/W7Nmz25R5yJAhWrZsmT7++GPFxcUpKChI/fr1a/LYF198UbNmzdLYsWN1zz33KDExURkZGVq0aNFJLwJ+6Pe//72+/fZbXXXVVRo2bJj8/Px04MABvfDCC8rPz9fTTz/dqnOc6fPcq1cvPf7443r44Ye1f/9+XXTRRQoLC1NOTo6+++47BQQE6LHHHtOWLVt0xx136Ec/+pH69u0rHx8fLV26VFu2bOFTcQBtRv32jPotSX/84x915ZVXavLkybrpppsUGxurPXv26A9/+IN8fHz029/+1nXsI488osOHD2vy5MlKSEhQUVGRnn/++UYzT++++2795z//0Xnnnad77rlHaWlpcjqdysjI0OLFi/XLX/5SY8aMaXXOXbt26cILL9TVV1+tCRMmKC4uToWFhfr000/197//XRMnTtT48eNdGd59911dcskl+vWvf62zzjpLlZWVWr58uWbOnKlJkybphhtu0Isvvqg5c+bo4MGDGjJkiL755hs98cQTmj59+innpzd4/vnndc455+jcc8/Vrbfeql69eqm0tFR79+7Vxx9/7Grgz5o1S4MHD9aoUaMUFRWlQ4cO6bnnnlNSUpL69u3b6ucCACRqtafU6tDQUH344YeaNWuWhg4dqltvvVVjx45VYGCg8vPz9fXXXys7O9tVA5titVr11FNP6dprr9XMmTN18803q7q6Wk8//bSKior0hz/84aSf+eCDD+Tl5aUpU6Zo+/bt+u1vf6uhQ4fqyiuvlCSNHz9eYWFhuuWWW/Too4/K29tbb7311kkfkrTGoEGDNHnyZE2bNk0pKSmqqqrSmjVr9MwzzygmJkY33nijpPpV+T4+Pvrxj3+s+++/X1VVVXrppZdOGkPTlLPPPls///nP9ZOf/ETr1q3Teeedp4CAAGVlZembb77RkCFDdOutt+qTTz7RX//6V1166aXq3bu3DMPQBx98oKKiIk2ZMqXNvyPcUOfvWwp8r2H37IYvHx8fIzo62pgwYYLxxBNPGLm5uSf9zIk7M69atcq47LLLjKSkJMNutxsRERHGhAkTjI8++qjRz33xxRfG8OHDDbvdbkgy5syZ0+jxjh07dtpzGUb9zuAzZsww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Lsy3zXl77j2VL+fhJr6Sj0/ny/n5VL6Yq5swZ3sGjUaDIUOGNPscACAzM9OrPgchBGbPno3PP/8cP/zwg9ferPKVcypfOX/w1VzpS/mhqXh++PBhfP/9917ZfeP06dOxe/fuZv+XYmNjsWDBAqxdu/aCt8MuXDxYYGDgGX2i+fv7Izw83Cv7SgOAP/7xj5g4cSISExNRXl6OZ599FrW1tV77eNCjjz6KESNGYPHixbjllluwdetWLF++HMuXL5c7tA7ldDqxYsUKzJgxAyqVd/8bmjhxIp577jkkJiYiPT0dO3fuxEsvvYR77rlH7tA6zNq1ayGEQK9evXDkyBEsWLAAvXr1wt133y13aO3CaDTiyJEj7u/z8/ORk5ODsLAwJCYmYt68eVi8eDHS0tKQlpaGxYsXw8/PD9OmTZMxajoXX8y3p/K13HsqX83DTXwpH5/OF/Pzqbw9Vzdhzu4a5s+fj+nTp2Po0KEYPnw4li9fDoPBgIceekju0NrNrFmz8NFHH+HLL79EYGCgu2VrcHAw9Hq9zNG1H185p/KV8wdvzpW+kh/OdZyxsbG46aabsGPHDnz99ddwOBzu/01hYWHQaDRyhd1q5/s8T78xoFarER0djV69el34TgR1KWPGjBFz586VO4wOc+utt4qYmBihVqtFbGysmDp1qti3b5/cYXWor776SmRkZAitVit69+4tli9fLndIHW7t2rUCgDh48KDcoXS42tpaMXfuXJGYmCh0Op3o0aOHWLhwobBYLHKH1mFWrlwpevToITQajYiOjhazZs0S1dXVcofVbjZs2CAAnDHNmDFDCCGE0+kUTz/9tIiOjhZarVaMHj1a7NmzR96gqdW8Pd+eyhdz76l8MQ838aV8fDpfzM+n8vZc3YQ5u+t4/fXXRVJSktBoNGLw4MEiKytL7pDaVUu/hwDEihUr5A6tw3nrOZUvnD94c670lfxwruPMz88/6/+mDRs2yB16q5zv8zxdUlKS+Oc//9mqfUhCCHHh5XYiIiIiIiIiIiIiIt/APtCJiIiIiIiIiIiIiFrAAjoRERERERERERERUQtYQCciIiIiIiIiIiIiagEL6ERERERERERERERELWABnYiIiIiIiIiIiIioBSygExERERERERERERG1gAV0IiIiIiIiIiIiIqIWsIBORERERERERERERNQCFtCJvNTGjRshSRKqq6vPuV737t3x8ssvd0pMRERERERERF2BJElYtWrVBa8/c+ZMTJky5aL2eezYMUiShJycnIvaTmuwJkB0fiygE3UBpaWlmDNnDnr06AGtVouEhARMnDgR69evP+t7RowYgZKSEgQHBwMA3n33XYSEhJyxXnZ2Nh544IF2ifPtt9/GgAED4O/vj5CQEAwaNAgvvPBCu2ybiIioK7niiiswb968M+avWrUKkiR1fkCNNmzYgLFjxyIsLAx+fn5IS0vDjBkzYLfbZYuJiIios5WWlmLu3LlITU2FTqdDVFQURo0ahbfeegsNDQ1yh3fBPvvsM1x66aUIDg5GYGAg0tPT8Yc//EHusIi8jkruAIjo3I4dO4aRI0ciJCQES5cuRf/+/WGz2bB27VrMmjULBw4cOOM9NpsNGo0G0dHR591+t27d2iXOd955B/Pnz8err76KMWPGwGKxYPfu3di/f3+7bL8lNpsNarW6w7ZPRETUlVitVmg0mrMu37dvHyZMmIBHHnkEr732GvR6PQ4fPoxPP/0UTqezQ2ISQsDhcECl4mUHERF5hqNHj7qvsRcvXox+/frBbrfj0KFD+Pe//43Y2FhMmjRJ7jDP6/vvv8dtt92GxYsXY9KkSZAkCfv37z9nQzsiahu2QCfycA8//DAkScLWrVtx0003oWfPnkhPT8f8+fOxZcsWAK5Hy9566y1MnjwZ/v7+ePbZZ5t14bJx40bcfffdqKmpgSRJkCQJixYtAnDm41rV1dV44IEHEBUVBZ1Oh4yMDHz99dfnjfOrr77CLbfcgnvvvRepqalIT0/H7bffjr/97W/N1vv3v/+N9PR0aLVaxMTEYPbs2e5lBoMBkydPRkBAAIKCgnDLLbegrKzMvXzRokUYOHAg/v3vf7tb4wshUFNTgwceeACRkZEICgrClVdeiV27dl3ET52IiKjj7dq1C2PHjkVgYCCCgoIwZMgQbNu2zb188+bNGD16NPR6PRISEvDII4+gvr7evbx79+549tlnMXPmTAQHB+P+++8/5/4yMzMRExODpUuXIiMjAykpKbj22mvxf//3f80K7z///DPGjBkDPz8/hIaG4pprrkFVVRUAwGKx4JFHHkFkZCR0Oh1GjRqF7Oxs93ubzj/Wrl2LoUOHQqvV4qeffoIQAkuXLkWPHj2g1+sxYMAAfPrpp+31oyQiIrpgDz/8MFQqFbZt24ZbbrkFffr0Qb9+/XDjjTdizZo1mDhxYovv27NnD6688kro9XqEh4fjgQcegNFoPGO9Z555xn1t+uCDD8JqtbqXfffddxg1ahRCQkIQHh6O3/3ud8jLy2vTcXz99dcYNWoUFixYgF69eqFnz56YMmUKXnvtNfc6eXl5mDx5MqKiohAQEIBLLrkE33///Tm3e77r6/OdvxB5IxbQiTxYZWUlvvvuO8yaNQv+/v5nLD+1S5ann34akydPxp49e3DPPfc0W2/EiBF4+eWXERQUhJKSEpSUlOCPf/zjGdtzOp2YMGECNm/ejA8//BD79+/H888/D6VSed5Yo6OjsWXLFhw/fvys67z55puYNWsWHnjgAezZswerV69GamoqAFcLtSlTpqCyshJZWVnIzMxEXl4ebr311mbbOHLkCP773//is88+c/cLd/3116O0tBTffPMNtm/fjsGDB+Oqq65CZWXleeMmIiKSyx133IH4+HhkZ2dj+/bteOKJJ9xPVu3ZswfXXHMNpk6dit27d2PlypXYtGlTsxvPAPDiiy8iIyMD27dvx5///Odz7i86OholJSX48ccfz7pOTk4OrrrqKqSnp+OXX37Bpk2bMHHiRDgcDgDAY489hs8++wzvvfceduzYgdTUVFxzzTVn5NzHHnsMS5YsQW5uLvr374+nnnoKK1aswJtvvol9+/bh0UcfxZ133omsrKy2/OiIiIja5OTJk1i3bt1Zr7EBtNjVWkNDA6699lqEhoYiOzsb//vf//D999+fkZfXr1+P3NxcbNiwAR9//DG++OILPPPMM+7l9fX1mD9/PrKzs7F+/XooFArccMMNbXoSLDo6Gvv27cPevXvPuo7RaMR1112H77//Hjt37sQ111yDiRMnwmAwtLi+EOK819fnOn8h8lqCiDzWr7/+KgCIzz///JzrARDz5s1rNm/Dhg0CgKiqqhJCCLFixQoRHBx8xnuTkpLEP//5TyGEEGvXrhUKhUIcPHiw1bEWFxeLyy67TAAQPXv2FDNmzBArV64UDofDvU5sbKxYuHBhi+9ft26dUCqVwmAwuOft27dPABBbt24VQgjx9NNPC7VaLcrLy93rrF+/XgQFBQmz2dxseykpKeLtt99u9XEQERG1hzFjxoi5c+eeMf+LL74QTafggYGB4t13323x/dOnTxcP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LzZs2IA5c+Zg+vTpiIqKcq9ntVpx7733Yv/+/e7W2bNnz4ZCoUBoaCjCw8OxfPlyHDlyBD/88APmz5/f5uNYtGgRHnvsMWzcuBH5+fnYuXMn7rnnHthsNowfPx6A6+b3559/jpycHOzatQvTpk0754Cl57u+Pt/5C5HXkrMDdiK6MMXFxWLWrFkiKSlJaDQaERcXJyZNmiQ2bNgghGg+uEmT0wcRFUKIhx56SISHhwsA4umnnxZCNB8wRAghTp48Ke6++24RHh4udDqdyMjIEF9//fV5Y/z000/FddddJ2JiYoRGoxGxsbHixhtvFLt372623ltvvSV69eol1Gq1iImJEXPmzHEvO378uJg0aZLw9/cXgYGB4uabb3YPyCKEaxDRAQMGnLHv2tpaMWfOHBEbGyvUarVISEgQd9xxR7MBSYmIiDrbtm3bxDXXXCMiIyNFUFCQGDp0qPj444+FEEJYLBZx2223iYSEBHfenD17tnuAUCGE2Lp1qxg/frwICAgQ/v7+on///uK5555zLz89h5/Pjh07xJ133imSk5OFVqsV4eHhYvTo0WL16tXN1tu4caMYMWKE0Gq1IiQkRFxzzTXu8wmTySTmzJkjIiIihFarFSNHjnQP9i1Ey+cfQgjhdDrFK6+84j4H6Natm7jmmmtEVlbWBcdPRETUXoqLi8Xs2bNFcnKyUKvVIiAgQAwbNky8+OKL7kFAT7/O3r17txg7dqzQ6XQiLCxM3H///aKurs69fMaMGWLy5MniL3/5iwgPDxcBAQHivvvuazYgZ2ZmpujTp4/QarWif//+YuPGjc3205pBRH/44Qdx4403us8loqKixLXXXit++ukn9zr5+fli7NixQq/Xi4SEBLFs2bIzBjo//XziXNfXF3L+QuSNJCEaOzMkIiIiIiIiIiIiIiI3duFCRERERERERERERNQCFtCJ6IJMmDABAQEBLU6LFy+WOzwiIiKft3jx4rPm6gkTJsgdHhEREbXCQw89dNa8/tBDD8kdHpFPYRcuRHRBioqKmo0mfqqwsDCEhYV1ckRERER0qsrKSlRWVra4TK/XIy4urpMjIiIiorYqLy9HbW1ti8uCgoIQGRnZyRER+S4W0ImIiIiIiIiIiIiIWsAuXIiIiIiIiIiIiIiIWsACOhERERERERERERFRC1hAJyIiIiIiIiIiIiJqAQvoREREREREREREREQtYAGdiIiIiIiIiIiIiKgFLKATEREREREREREREbWABXQiIiIiIiIiIiIiohawgE5ERERERERERERE1IL/B6umrOvoWidpAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 1500x500 with 3 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(15, 5))  # Create a figure with specified size\n",
    "\n",
    "\n",
    "# Create histograms for Critic Score, User Score, and Global Sales\n",
    "plt.subplot(131)\n",
    "sns.histplot(df_cleaned['Critic_Score'], kde=True)\n",
    "plt.title('Distribution of Critic Scores')\n",
    "\n",
    "plt.subplot(132)\n",
    "sns.histplot(df_cleaned['User_Score'], kde=True)\n",
    "plt.title('Distribution of User Scores')\n",
    "\n",
    "plt.subplot(133)\n",
    "sns.histplot(df_cleaned['Global_Sales'], kde=True)\n",
    "plt.title('Distribution of Global Sales')\n",
    "\n",
    "plt.tight_layout() # Adjust the layout to prevent overlap\n",
    "plt.show() # Display the plot"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "04033b90-20da-4386-a7f9-18fde28616ca",
   "metadata": {},
   "source": [
    "### Set 2: Basic Grouping and Analysis\n",
    "\n",
    "##### Step 1: Top Genres by Sales"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "13149345-326c-47a1-98f7-09a4a9c9c4cc",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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CxUqpI0aNdIKFSromjVrjOBeokQJnTp1qnHHhZkzZ3If9jcYPny4pk6dWn/77Tc9ePCgNmrUSBMlSqQ3btzQO3fu6MqVK7VatWparVo1bd68udHLo379+tq3b9/XzpYdV0Suu3///po1a1ZNly6dpk6dWtu0aaMPHjxQVdVp06Zp+fLltU2bNqbpjh1fHT58WEuXLm1Mcmx19uxZbdOmjWbPnl1/+OEHY7mjGtMiv++3bNliU6/1WN2gQQPt06ePsbxfv342vVMQcfEi8uR9M2bMUNWI41LGjBl19OjRGhwcrI8ePdJFixZp0aJFtXTp0lq9evV31tOU0O4Af/31l44cOVJfvnxpjDm5f/++3r59WwMCAjRVqlQ6cuRIY/sNGzbo1KlT9Ycffkiw94O2fpBWrFihhQsX1qxZs2quXLlsuq388MMP6uTkpE2bNtUuXbposmTJ6H74GpFPDl68eKF79uzRmjVratWqVW1uh9evXz8tWrSoDh8+XAMDA9/4HHGJte5z587pH3/8YROw1q1bpyVLltQmTZrEyq06zM765R75BOD+/ft68eJFLV68uNapU0dXr15t8ztHjhxRd3d3zZcvn/7444/G78cXf/zxh/HvgIAAXb58uXG7tmLFiuno0aON4L5jxw7NmTOn+vv7x+vjTuTA/vvvvxvHhm3btqmTk5O2a9fOJri3atVKLRbLay9mxDWvXqywhsa1a9dqu3btdP369dqwYUMtU6aMrl27Vp8/f64dOnTQ0qVL64QJE+LVhU57u3PnjlapUkXXrl2rqhETrKVMmdI4cY4scoui9U4n8WVC3m+//VbTpEmju3bt0oMHD+r69es1TZo0WrNmTeP9891332mePHleOywQMWfdr5Evrt6/f1+vXLmitWrV0kqVKkXphffHH3+ou7u7enl56ZQpU95pva96ddK5nDlz6t27d22+iz/77DNt0qSJqqoOHjxYkyRJEmVIV0IWEhKipUqV0jJlyujChQvVYrEYc2WpRvQAsgb3yA0TL1++jLVJ516H0O4A3333nbq6umrjxo01WbJkxpgT1YjxbtbgPmrUKAdWaT4bN25UZ2dnDQgI0PXr12uvXr20WLFiNq07a9eu1bJly2rt2rXjxURPsWnGjBlGr44tW7ZovXr1tFKlSjbBvX///urr62tzNTkuijz2b8mSJZo1a1ZN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N15/fD3evGuRe7/27NlTP//8cyOwqUbcLSRRokQ6btw4vXLlio4cOVKzZMmiwcHBpji+bdq0SatVq6YVKlSIMmluzZo1dcSIETp8+HBNlixZvLhzh71E/n+3cuVKdXV11VGjRumiRYu0efPm6uLiouvXr1fViB4Y+/fv1zZt2miTJk306NGjjirbQGh/R2bPnq2VK1fWp0+fvvb2EqoRrTsbNmyId11OY2rDhg3q7OysEyZM0FmzZum4ceM0WbJk2rlzZ1WN+NANHz5cP/nkE/30008TZOvoP/m7sD1r1ixt37690d23aNGimitXLmNcoRm+kP4Na4hYsWKFWiwWTZcuna5fv94IXpcvX9aOHTtqzpw5NW3atFq2bFlNkyZNvJqBOCbWrVuniRIl0po1a6qrq6uWKlVK582bZ5wcjh07Vi0WixYvXlxTpkwZ72adrVSpkqZNm1ZXr16tmzdv1kKFCum2bdv0xo0b2r17d02TJo1Onz7deP8cOnRI27Ztq/369TNavuJDoIgs8vHjxx9/1C5duugnn3xic//frl27arJkyfTDDz/UVq1aably5bRAgQIJqjXw7NmzWqNGDa1evbru2LHD0eWY0qsXfyK3XIaGhmqLFi3U3d1dt23bZiz//PPP49297K3fS6oRn6lhw4bpoEGDbLZZtmyZlipVSr/66qvXfn7i23HG0X777TdNnDixNmzYUD09PdXPz89mjp8xY8aoi4uLFilSxKETuL3pWLp161atX7++lilTxubz07JlS7VYLJoiRQoC+/97tZfE+fPntUyZMjpp0iRVVb127ZpmyZJFs2bNqk5OTsbQSqt3eVu3v0Not6NXg1Lkx6NHj9YMGTIYXbhf7ZqLCOHh4dq9e3dt166dzfINGzZo0qRJ9auvvoqyPWxFft9t27ZNFyxYoLt3747Sirxx40YdNmyYFihQQC0Wi3700Udx8iq+9fWGhIQYy06dOqW7du3SefPmaaNGjTRbtmy6bNkyI3jdu3dPT5w4oWPHjtW5c+fq+fPnHVK7o0QegtKxY0fjXsghISFaq1YtLV26tP7yyy/G+2Hz5s36yy+/GBd24oPInxPr5HojR46MciLdq1cvI7i/7pZdcfEzE139+/dXLy8v7d69uzZq1EgLFSqkn3/+ubF++vTp+vHHH2uTJk10yJAh8bYb8985e/as1q5d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o0ZpwYIFjV4HvXv31tSpU+uIESP06tWr+tdff+mAAQM0derUNuO444Pp06ers7OzcfuyqlWraqlSpYz70d+4cUNr166trq6u+u233+rMmTPV399fy5cvr6qqhQoV0r59+zqs/nftr7/+0urVq+sHH3xgM/nRq62ghAuo/i/0lC5dWj09PW0myLL67bfftGbNmlqqVCk9ceKEqtJCePjwYa1du7b26tXL+GwlxJ4GsSk8PFy///57LVCggJ4/f17DwsJ08ODBmj17du3du7cGBgbqiRMndODAgerh4REnvvvOnj2rtWvX1lKlStncBhlvNmjQIC1SpIgOHTrUpsV99uzZmi9fPm3QoIHeuHHDgRW+HUJ7LLp27ZpWrFgxwbYYv3jxQsuVK2fMjqoa8UXeu3dvTZUqlWbMmFEbN26sqhHB3c3NjQlY/sby5cs1T5482qlTJ5sLQWPHjo1yu49nz57p3LlztXHjxnHyQkjkml++fKmjRo3SRIkS6U8//aSq/zv5O3v2rObJk8foOnblyhWtV6+eDh06VK9cufLuCzeRu3fvqqenp01vn+7du2vhwoXVyclJixcvrnny5NHDhw87rshYMGPGDE2aNKnNWPYHDx5orVq1tEaNGvrzzz+rqmpwcLD2799fc+bMqQULFtTq1asbk18WL17c5riVEFi7yteoUYMWHbzR+vXrddSoUXr9+nUdMWKEWiwWzZQpkx4+fDhKAP3tt9+0du3ami1btjjVBTU2PXjwgDsvxLKrV69qxowZdejQoaoacS46cOBALVasmCZJkkTff/99LVCgQJz67jt16pQ2btzYVOPuzcD6WTp9+rSeOHHC5i5T/fv31yJFiuiQIUOM8+PBgwfrhAkTTDsk4p8Q2mOZ9VZbCVFQUJARMk+dOqUBAQGaO3dubdSokX777bf6448/aubMmXXEiBGqqlqlShUmYPkHS5Ys0WLFimnLli31ypUrun79enV2djZu2RS5JSPybWPiUnC31nr58mXjnuLLli3TqVOn2gR31Ygvsrx58+ro0aP1zp07OmzYMPX399f79+87qvx3LjQ0NMr/X+v7YPLkyVqiRAmbW5mdP39e16xZo0eO/F979x5Q8/3/Afx5zunmkrbphpWmspLWRRQZipAs0himzMiabGbf6Ouy+KK+krutbQi5bOTWNiIrakr0LXdJyW1G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",
      "text/plain": [
       "<Figure size 1200x600 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Group data by Genre and calculate total Global Sales for each genre\n",
    "genre_sales = df_cleaned.groupby('Genre')['Global_Sales'].sum().sort_values(ascending=False)\n",
    "\n",
    "# Create a bar plot of total sales by genre\n",
    "plt.figure(figsize=(12, 6))\n",
    "sns.barplot(x=genre_sales.index, y=genre_sales.values)\n",
    "plt.title('Total Global Sales by Genre')\n",
    "plt.xlabel('Genre')\n",
    "plt.ylabel('Total Global Sales (millions)')\n",
    "plt.xticks(rotation=45)   # Rotate x-axis labels for better readability\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ada5e9e3-fc68-484a-9c95-8135cb826af8",
   "metadata": {},
   "source": [
    "##### Step 2: Correlation between Scores and Sales"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "1121fb76-062d-431a-9ab1-c511974dfabb",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Correlation between Critic Score and Global Sales: 0.50\n"
     ]
    }
   ],
   "source": [
    "# Calculate the correlation coefficient between Critic Score and Global Sales\n",
    "correlation = df_cleaned['Critic_Score'].corr(df_cleaned['Global_Sales'])\n",
    "print(f\"Correlation between Critic Score and Global Sales: {correlation:.2f}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0e66f1ad-c269-45e0-b5e4-d2b0a0571835",
   "metadata": {},
   "source": [
    "##### Step 3: Sales Over Time"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "461ada93-8cfe-41f2-ae06-2998a571834c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAA9wAAAIhCAYAAAC8K7JuAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy81sbWrAAAACXBIWXMAAA9hAAAPYQGoP6dpAAChNUlEQVR4nOzdd3iTZdsG8PPJ7N4p0JYOaNmrlA2CKCgCKi6UJUtFREHAgfvVT0FUFFkivggyHbhQRF9ERPYqe7alE+gu3SvJ8/2RJlBbIIGkT8b5O44e0KcZZ4GWXrnu+7oFURRFEBEREREREZFVyaQOQEREREREROSMWHATERERERER2QALbiIiIiIiIiIbYMFNREREREREZAMsuImIiIiIiIhsgAU3ERERERERkQ2w4CYiIiIiIiKyARbcRERERERERDbAgpuIiIiIiIjIBlhwExG5oAULFkAQBLRr107qKHZJr9djzZo1uPvuuxEcHAylUgk/Pz/06NEDH330EXJzc2vdPjIyEuPGjbP4eVJSUiAIAj766CMrJb/ymCtXrrzhbdPT0/HMM8+gRYsWcHd3R0BAANq3b48nn3wS6enpFj/333//DUEQ8Pfff1se/BadPn0a48aNQ3h4OFQqFYKCgjB48GBs3ry5wbNcz7hx4yAIwg3fxo0bJ+mfJxERWYdC6gBERNTwvvzySwDAyZMnsW/fPnTv3l3iRPajvLwc999/P/788088+uijWLBgAUJCQlBUVITdu3fjww8/xM8//4wdO3ZIHfWWZGRkoHPnzvDz88PMmTPRsmVLFBYW4tSpU/j2229x/vx5NG3aVOqYZvnhhx8wcuRINGvWDG+88QZatmyJrKwsrFixAoMHD8aLL76IDz74QOqYAIA33ngDTz/9tOn9+Ph4TJkyBbNnz0b//v1N1zUaDTQaDfbs2YM2bdpIEZWIiKyABTcRkYs5ePAgjh49iiFDhmDTpk1Yvnx5gxfcoiiioqIC7u7uDfq85nj++eexZcsWrFu3DiNGjKj1saFDh+L111/H2rVrJUpnPV988QVyc3Oxf/9+REVFma4PGzYMr776KvR6vYTpzJeUlIQxY8agffv2+Pvvv+Hp6Wn62COPPILJkyfjww8/ROfOnfHYY481WK7q6moIggCFovaPWs2bN0fz5s1N71dUVAAAYmJi0KNHjzqPU981IiJyHFxSTkTkYpYvXw4AeP/999GrVy98/fXXKCsrA2AoEoKDgzFmzJg697t8+TLc3d0xY8YM07WioiK88MILiIqKgkqlQmhoKJ5//nmUlpbWuq8gCHj22WexdOlStG7dGmq1Gl999RUA4O2330b37t0REBAAHx8fdO7cGcuXL4coirUeo7KyEjNnzkTjxo3h4eGBvn374tChQ/Uu587MzMSkSZMQFhYGlUqFqKgovP3229Bqtdf9s7l06RK+/PJLDBkypE6xbeTh4YEnn3zyuo8DAGlpaRg9ejSCg4OhVqvRunVrzJs3r95CVq/X47333kN4eDjc3NzQpUsXbN26tdZtEhMTMX78eMTExMDDwwOhoaG49957cfz48RtmqU9eXh5kMhmCg4Pr/bhMduVHhIMHD+Kxxx5DZGQk3N3dERkZiREjRiA1NdWs5zp48CDuu+8+BAQEwM3NDbGxsfj2229r3aasrMz0b8nNzQ0BAQHo0qUL1q9ff93H/uSTT1BWVoaFCxfWKraN5s2bBz8/P7z33nsAgKNHj0IQBNPXwdU2b94MQRCwceNG07WEhASMHDmy1t/j4sWLa93PuPR79erVmDlzJkJDQ6FWq5GYmGjWn8+11LekfNy4cfDy8sKZM2dw9913w9PTE02aNMH7778PANi7dy/69OkDT09PtGjRwvR1drWb/fogIiLLscNNRORCysvLsX79enTt2hXt2rXDhAkT8MQTT+C7777D2LFjoVQqMXr0aCxduhSLFy+Gj4+P6b7r169HRUUFxo8fD8BQIPXr1w8ZGRl49dVX0aFDB5w8eRJvvvkmjh8/jj///BOCIJju/9NPP2HHjh1488030bhxY1Ohl5KSgkmTJiE8PByAoWB47rnncOHCBbz55pum+48fPx7ffPMNXnrpJdxxxx04deoUHnjgARQVFdX6HDMzM9GtWzfIZDK8+eabaN68Ofbs2YN3330XKSkpWLFixTX/fLZt2watVov77rvvlv6cc3Jy0KtXL1RVVeH//u//EBkZiV9//RUvvPACkpKSsGTJklq3X7RoESIiIjB//nzo9Xp88MEHuOeee7B9+3b07NkTAHDx4kUEBgbi/fffh0ajQX5+Pr766it0794dhw8fRsuWLS3K2LNnTyxevBgPPvggZsyYgZ49e9b6+75aSkoKWrZsicceewwBAQG4dOkSPvvs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Xl4ePvzwQwCAj48P+vbtiw8//BBBQUGIjIzE9u3bsXz5crM6659//jnuuece3H333Rg3bhxCQ0ORn5+P06dPIz4+Ht999x0AoHv37hg6dCg6dOgAf39/nD59GqtXr0bPnj2vWWwDhnOtV69ejVGjRqFr166YMWMGWrZsiaysLHz55ZfYvHkzXnjhBTz66KO17ieXy/H444/j448/ho+PDx588EH4+vrWus2nn36KPn364LbbbsPkyZMRGRmJ4uJiJCYm4pdffqmzF98R3OrXBxERWUjKiW1ERNQwhg0bJqpUKjE7O/uat3nsscdEhUIhZmZmiqJomBjetGlTEYD42muv1XufkpIS8fXXXxdbtmwpqlQq0dfXV2zfvr04ffp00+OIomFK+ZQpU+p9jC+//FJs2bKlqFarxWbNmolz5swRly9fLgKoNS27oqJCnDFjhhgcHGyaTL1nzx7R19e31pRuURTFnJwccerUqWJUVJSoVCrFgIAAMS4uTnzttdfEkpKSG/556XQ6cdWqVeLAgQPFoKAgUaFQiL6+vmK3bt3EN954Q8zIyKh1+39PKRdFUUxNTRVHjhwpBgYGikqlUmzZsqX44YcfmqZyi+KVieJz584V3377bTEsLExUqVRibGys+Mcff9R6vIKCAnHixIlicHCw6OHhIfbp00fcsWOH2K9fP7Ffv351HvNGU8r37t0rTpkyRezYsaMYEBAgyuVyUaPRiIMGDao1YV4URTEjI0N86KGHRH9/f9Hb21scNGiQeOLEiTqfd31TtUVRFI8ePSoOHz5cDA4OFpVKpdi4cWPxjjvuME3LF0XDZPcuXbqI/v7+pn8L06dPF3Nzc6/7eRidPHlSHDt2rBgWFmb6Ox80aJC4adOma97n3LlzIgARgLhly5Z6b5OcnCxOmDBBDA0NFZVKpajRaMRevXqJ7777bp3Pu75J4zdyM1PKPT0969y2X79+Ytu2betcj4iIEIcMGVLr2q1+fRARkfkEUawZ3UlERORgdu/ejd69e2Pt2rUYOXKk1HGIiIiIamHBTUREDmHLli3Ys2cP4uLi4O7ujqNHj+L999+Hr68vjh07VmdqMxEREZHUuIebiIgcgo+PD/73v/9h/vz5KC4uRlBQEO655x7MmTOHxTYRERHZJXa4iYiIiIiIiGxAduObEBEREREREZGlWHATERERERER2QALbiIiIiIiIiIbcOihaXq9HhcvXoS3tzcEQZA6DhERERERETk5URRRXFyMkJAQyGTX72E7dMF98eJFNG3aVOoYRERERERE5GLS09MRFhZ23ds4dMHt7e0NwPCJ+vj4SJyGiIiIiIiInF1RURGaNm1qqkevx6ELbuMych8fHxbcRERERERE1GDM2dbMoWlERERERERENsCCm4iIiIiIiMgGWHATERERERER2QALbiIiIiIiIiIbYMFNREREREREZAMsuImIiIiIiIhsgAU3ERERERERkQ2w4CYiIiIiIiKyARbcRERERERERDbAgpuIiIiIiIjIBlhwExEREREREdkAC24iIiIiIiIiG2DBTURERERERGQDLLiJiIiIiIiIbIAFNxEREREREZENsOAmIiIiIiIisgEW3EREREREdF05xZU4falI6hhEDocFNxERERERXdOWU1m446O/MXjBDmQUlEkdh8ihKKQOQERERERE9kevFzF/awIWbE0wXUvKKUWYv4eEqYgcCwtuIiIiIiKqpbCsGs9/cxjbzuYAADxUcpRV6ZBTXClxMiLHwiXlRERERERkcvpSEe5dtBPbzubATSnDJ492xN1tGwMAcktYcBNZgh1uIiIiIiICAPx85AJmfX8c5dU6hPm74/MxcWgb4ovTl4oBALnscBNZhAU3EREREZGLq9bp8f7mM1i+MxkA0LeFBgse6wQ/DxUAIMjL8GsOO9xEFmHBTURERETkwnKKK/HsunjsS84HADzbPxrTB7aAXCaYbqPxVgPgknIiS7HgJiIiIiJyUYfTCjB5TTwyiyrgpVbgo0c6YlC7xnVuF+RlKLg5NI3IMiy4iYiIiIhc0Lp9afjPxpOo0unRXOOJz8d0QXSwV723NRbcuSVVDRmRyOGx4CYiIiIiciEV1Tr8Z+NJfH0gHQAwqG1jfDS8I7zU1y4NjEvK80urUK3TQynnYUdE5mDBTURERETkIi5eLsfkNYdwNKMQMgF44e6WmNyvOQRBuO79/D1UkAmAXjQU3Y183BooMZFjY8FNREREROQCdifl4rl1h5FXWgU/DyUWPBaLvi00Zt1XLhMQ6KVGTnElcoorWXATmYkFNxERERGRExNFEct3JmPO5jPQ6UW0DfHB0tFxaBrgYdHjBBkLbk4qJzIbC24iIiIiIidVVqXFSxuO4ddjlwAAD3YO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",
      "text/plain": [
       "<Figure size 1200x600 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Calculate average Global Sales for each year\n",
    "yearly_sales = df_cleaned.groupby('Year')['Global_Sales'].mean()\n",
    "\n",
    "# Create a line plot of average sales over time\n",
    "plt.figure(figsize=(12, 6))\n",
    "plt.plot(yearly_sales.index, yearly_sales.values)\n",
    "plt.title('Average Global Sales Over Time')\n",
    "plt.xlabel('Year')\n",
    "plt.ylabel('Average Global Sales (millions)')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "689d0893-0171-4b17-99ef-efd7758a1a77",
   "metadata": {},
   "source": [
    "#### Set 3: Advanced Analysis and Visualization\n",
    "\n",
    "##### Step 1: Platform Comparison (for a specific genre, e.g., 'Action')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "9e952476-64a0-44a5-97c1-60f4314996ba",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAA+UAAAI4CAYAAAAMFEjPAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy81sbWrAAAACXBIWXMAAA9hAAAPYQGoP6dpAABkNElEQVR4nO3de3yO9ePH8fe9031vY2NzNjbnnA+JHLJRWk6RJFJCSU45RKJyKIcUpRMhoQPRAYUaYSgqSgc5VHIqpIQxNmyf3x99d//cbLObe7vuba/n47HHw31d133d7/u+7l32vq/PfV02Y4wRAAAAAADIcT5WBwAAAAAAIL+ilAMAAAAAYBFKOQAAAAAAFqGUAwAAAABgEUo5AAAAAAAWoZQDAAAAAGARSjkAAAAAABahlAMAAAAAYBFKOQAAAAAAFqGUA/BaP/74ox544AFVqFBBgYGBCgwMVKVKldSnTx9t3brVZdmxY8fKZrNd1ePExMSoRo0anojsss6YmJgrLnf+/HnNnDlTN9xwg8LCwhQUFKTIyEi1b99eS5YsuarHjoqKUo8ePa7qvlcrKipKNptNNptNPj4+Cg0NVdWqVdW9e3etWrUq3fvYbDaNHTvWrcdZuXKl2/dJ77HmzZsnm8122fvoWhw6dEhjx47V999/f9m8a3l/Xqt///1XXbp0UbFixWSz2dShQ4cce+x69erJZrNpypQpV72OzLa5Fe/1i+3du1ePPPKIqlatquDgYDkcDkVFRenee+/VunXrZIyxLFt2yo59Znou3q/YbDYVKFBADRs21FtvvXVZnqzsb9Mzffp0zZs3L91527ZtU3R0tEJDQ2Wz2TRt2rSregwAuBI/qwMAQHpmzpypAQMGqEqVKho0aJCqV68um82mnTt3auHChbrhhhv022+/qUKFClZHvSb33XefPvroIw0ePFjjxo2T3W7X77//rs8++0xxcXG64447rI6YZU2aNHGWr9OnT2v37t167733FBsbqzvvvFMLFy6Uv7+/c/nNmzcrIiLCrcdYuXKlXnvtNbeL+dU8lrsOHTqkcePGKSoqSnXq1HGZ9+CDD+q2227L1sfPyDPPPKMlS5bozTffVIUKFRQWFpYjj/v9999r27ZtkqQ5c+Zo2LBhV7WezLb5kiVLFBISci0xr9rHH3+se+65R0WKFNHDDz+sevXqyW6367ffftMHH3ygFi1a6PPPP9fNN99sSb684uL9yh9//KEpU6bo/vvvV2Jiovr27XvN658+fbqKFCmS7oc7vXr1UmJiot577z0VLlxYUVFR1/x4AJAeSjkAr/Pll1+qX79+atOmjT744AMFBAQ457Vo0UL9+/fX+++/r8DAQAtTXru9e/dq0aJFGj16tMaNG+ecfvPNN6t3795KTU21MJ37ChUqpBtvvNF5+5ZbblH//v01duxYjRs3Tk8++aQmT57snH/xstnBGKOkpCQFBgZm+2NdSURERLZ/KJCR7du3q0KFCurWrZtH1nfx65qZN954Q5LUpk0brVixQps2bVLjxo09kiFN3bp1Pbq+rNqzZ4+6du2q6tWr6/PPP3f5YCA6OloPPPCA4uPjVbhwYUvy5SXp7VciIyP1wgsveKSUZ2b79u3q3bu3WrVq5ZH1nT9/XjabTX5+/PkNwBXD1wF4nYkTJ8rX11czZ850KeQXu+uuu1SqVKlM15OamqrnnntO1113nex2u4oVK6bu3bvrjz/+SHf5jRs36sYbb1RgYKBKly6tp556SikpKS7LjBs3Tg0bNlRYWJhCQkJUr149zZkz56qGqR47dkySVLJkyXTn+/j8/y46KSlJjz76qOrUqaPQ0FCFhYWpUaNGWrZsWZYeKyEhQcOGDVO5cuUUEBCg0qVLa/DgwUpMTHRZ7v3331fDhg0VGhqqoKAglS9fXr169XL7uV1s7Nixql69ul599VUlJSU5p186pPzMmTPOjA6HQ2FhYapfv74WLlwoSerRo4dee+01533Tfvbt2+ecNmDAAL3++uuqWrWq7Ha75s+fn+5jpTl+/Lh69uypsLAwBQcHq127dvr9999dlsloiPTFQ2bj4+N1ww03SJJ69uzpzJb2mOkNX8/q+zNtqPCWLVt00003ObfLs88+m+kHN/v27ZPNZtPnn3+unTt3OjPFx8dL+m9Ye79+/VS6dGkFBASofPnyeuKJJ5ScnOyynsxe14wkJSVpwYIFuv766/Xiiy9Kkt588810l/3ss8908803O99zVatW1aRJkyRdeZunt20OHDige++9V8WKFZPdblfVqlU1depUl9cq7bWZMmWKXnjhBZUrV04FChRQo0aN9NVXX2X63CTphRde0JkzZzR9+vQMj9THxMSodu3aztu//fabevbsqUqVKikoKEilS5dWu3bt9NNPP7ncLz4+XjabTQsWLNCIESNUsmRJFShQQO3atdNff/2lU6dO6aGHHlKRIkVUpEgR9ezZU6dPn3ZZhzFG06dPV506dRQYGKjChQurU6dOl723t23bprZt2zpfq1KlSqlNmzYZ7iMvldk+0xijSpUqKTY29rL7nT59WqGhoerfv3+WHudihQoVUpUqVbR///5Ml8vKvjoqKko///yz1q9f73xvRUVFOb/ecuHCBc2YMcM5L8327dvVvn17FS5cWA6HQ3Xq1LnsdyJtO7799tt69NFHVbp0aedIih49eqhAgQLatWuXYmNjFRwcrJIlS+rZZ5+VJH311Vdq2rSpgoODVbly5Sv+vgHI/fioDoBXSUlJ0bp161S/fv0My2pW9e3bV7NmzdKAAQPUtm1b7du3T0899ZTi4+P13XffqUiRIs5ljxw5oi5duujxxx/X008/rRUrVmj8+PE6fvy4Xn31Vedy+/btU58+fVS2bFlJ//3xNHDgQP35558aPXq0W/mqVq2qQoUKady4cfLx8dGtt96a4fDI5ORk/fvvvxo2bJhKly6tc+fO6fPPP1fHjh01d+5cde/ePcPHOXPmjKKjo/XHH39o1KhRqlWrln7++WeNHj1aP/30kz7//HPZbDZt3rxZd999t+6++26NHTtWDodD+/fv19q1a916Xulp166dnn32WW3dulVNmzZNd5mhQ4fq7bff1vjx41W3bl0lJiZq+/btzg8vnnrqKSUmJuqDDz7Q5s2bnfe7+H2ydOlSbdy4UaNHj1aJEiVUrFixTHM98MADatmypRYsWKCDBw/qySefVExMjH788UcVKlQoy8+vXr16mjt3rnr27Kknn3xSbdq0kaRMj467+/7s1q2bHn30UY0ZM0ZLlizRyJEjVapUqQy3fcmSJbV582b169dPJ0+e1LvvvitJqlatmpKSktS8eXPt2bNH48aNU61atbRx40ZNmjRJ33//vVasWOGyLndf148++kjHjx9Xr169VKlSJTVt2lSLFi3StGnTVKBAAedyc+bMUe/evRUdHa3XX39dxYoV0y+//KLt27dLyto2v9jff/+txo0b69y5c3rmmWcUFRWl5cuXa9iwYdqzZ4+mT5/usvxrr72m6667zvld4aeeekqtW7fW3r17FRoamuHzW716tUqWLKn69etn+jpc7NChQwoPD9ezzz6rokWL6t9//9X8+fPVsGFDbdu2TVWqVHFZftSoUWrevLnmzZunffv2adiwYeratav8/PxUu3ZtLVy4UNu2bdOoUaNUsGBBvfzyy8779unTR/PmzdMjjzyiyZMn699//9XTTz+txo0b64cfflDx4sWVmJioli1bqly5cnrttddUvHhxHTlyROvWrdOpU6eu+HyutM+02WwaOHCgBg8erF9//VWVKlVy3vett95SQkLCVZXy8+fPa//+/SpatGimy2VlX71kyRJ16tRJoaGhzveG3W5XRESENm/erEaNGqlTp0569NFHnevdvXu3GjdurGLFiunll19WeHi43nnnHfXo0UN//fWXHnvsMZccI0eOVKNGjfT666/Lx8fH+btz/vx5dezYUQ8//LCGDx+uBQsWaOTIkUpISNCHH36oESNGKCIiQq+88op69OihGjVq6Prrr3f79QKQSxgA8CJHjhwxkkyXLl0um3fhwgVz/vx5509qaqpz3pgxY8zFu7SdO3caSaZfv34u6/j666+NJDNq1CjntOjoaCPJLFu2zGXZ3r17Gx8fH7N///50s6akpJjz58+bp59+2oSHh7vkiY6ONtHR0Vd8vitWrDBFihQxkowkEx4ebu666y7z8ccfZ3q/tNfigQceMHXr1nWZFxkZae6//37n7UmTJhkfHx+zZcsWl+U++OADI8msXLnSGGPMlClTjCRz4sSJK+a+VGRkpGnTpk2G82fMmGEkmUWLFjmnSTJjxoxx3q5Ro4bp0KFDpo/Tv39/k9F/XZJMaGio+ffff9Odd/FjzZ0710gyd9xxh8tyX375pZFkxo8f7/LcLn4901y6jbds2WIkmblz5162rCfen19//bXLstWqVTOxsbGXPVZ6OatXr+4y7fXXXzeSzOLFi12mT5482Ugyq1atck7L7HXNSIsWLYzD4TDHjx83xvz/6z1nzhznMqdOnTIhISGmadOmLr87l8psm1+6bR5//PF0X6u+ffsam81mdu/ebYwxZu/evUaSqVmzprlw4YJzuW+++cZIMgsXLsz0+TkcDnPjjTdeNj1tn5D2k5KSkuE6Lly4YM6dO2cqVapkhgwZ4py+bt06I8m0a9fOZfnBgwcbSeaRRx5xmd6hQwcTFhbmvL1582YjyUydOtVluYMHD5rAwEDz2GOPGWOM2bp1q5Fkli5dmulzTU9W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0VtvvaU2bdro6aefVmRkpFasWKHp06erb9++qly58lU9r0tVqFBBCxcuVNeuXVWzZk317dtX9erVk91u19GjR7Vq1SpJrvuotm3bat68ebruuutUq1Ytffvtt3r++ec9fv36Jk2a6KGHHlLPnj21detWNWvWTMHBwTp8+LC++OILZ97ly5dr+vTp6tChg8qXLy9jjD766COdOHFCLVu2vOLjZGWfmaZly5aqVq2a1q1b57xcXXZyZ1+dNlpg0aJFKl++vBwOR7ofXKQZM2aMli9frubNm2v06NEKCwvTu+++qxUrVui5557L9Kz9AJAhi04wBwBX9P3335uePXuacuXKGbvdbhwOh6lYsaLp3r27WbNmjcuyl57d2pj/zrg7efJkU7lyZePv72+KFCli7r33XnPw4EGX5dLOTh0fH2/q169v7Ha7KVmypBk1apQ5f/68y7JvvvmmqVKlirHb7aZ8+fJm0qRJZs6cOUaS2bt3r8s6r3T29ePHj5vx48ebFi1amNKlS5uAgAATHBxs6tSpY8aPH2/OnDnjsvyzzz5roqKijN1uN1WrVjWzZ89O93mnd7bw06dPmyeffNJUqVLFBAQEmNDQUFOzZk0zZMgQc+TIEWOMMcuXLzetWrVyZilWrJhp3bq12bhxY6bPI+0x9b8zyNtsNlOgQAFTpUoVc99995m4uLh076NLzoj++OOPm/r165vChQs7X98hQ4aYf/75x7lMcnKyefDBB03RokWNzWZzed2VzpmSM3qstLOBr1q1ytx3332mUKFCJjAw0LRu3dr8+uuvLvdNTU01zz33nClfvrxxOBymfv36Zu3atelu44ULF5rrrrvO+Pv7uzymJ96fl7r//vtNZGRkus83K/c/duyYefjhh03JkiWNn5+fiYyMNCNHjjRJSUkuy2X2ul7s77//NgEBAZmeQf/48eMmMDDQ5cziK1euNNHR0SY4ONgEBQWZatWqmcmTJzvnZ7bN03uv79+/39xzzz0mPDzc+Pv7mypVqpjnn3/e5UzoaWdff/755y/LeOl7JTN79uwxAwcONFWqVDGBgYHGbrebyMhIc9ddd5klS5a4nOX7+PHj5oEHHjDFihUzQUFBpmnTpmbjxo2XvY/Sztr9/vvvuzxW2nv20qsopL23/v77b5fpb775pmnYsKEJDg42gYGBpkKFCqZ79+5m69at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",
      "text/plain": [
       "<Figure size 1200x600 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Filter the dataset for Action games\n",
    "action_games = df_cleaned[df_cleaned['Genre'] == 'Action']\n",
    "\n",
    "# Create a box plot to compare Global Sales across different platforms for Action games\n",
    "plt.figure(figsize=(12, 6))\n",
    "sns.boxplot(x='Platform', y='Global_Sales', data=action_games)\n",
    "plt.title('Global Sales Distribution for Action Games by Platform')\n",
    "plt.xlabel('Platform')\n",
    "plt.ylabel('Global Sales (millions)')\n",
    "plt.xticks(rotation=90)  # Rotate x-axis labels for better readability\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "310c4527-f0f2-4d79-887b-3af51eb9aee6",
   "metadata": {},
   "source": [
    "##### Step 2: Predicting Sales"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "c262c7d5-8a95-4a17-8ef9-c026da01f11c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Mean Squared Error: 2.85\n",
      "R-squared: 0.37\n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAA0EAAAIhCAYAAACIfrE3AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy81sbWrAAAACXBIWXMAAA9hAAAPYQGoP6dpAACA3klEQVR4nO3dZ3RU1duG8WtSJx0IBAiE3juIICBSRBFQwIrYABsoIkVFwUKRolgQ/goKUsQCVuw0FVRQpInSpDdJILRUSEKS8344bzIMBMyknUzm/q01i8ye9mQgIXf23s+2GYZhICIiIiIi4iG8rC5ARERERESkKCkEiYiIiIiIR1EIEhERERERj6IQJCIiIiIiHkUhSEREREREPIpCkIiIiIiIeBSFIBERERER8SgKQSIiIiIi4lEUgkRERERExKMoBImIW5s+fTo2m41GjRrl+Tmio6MZO3YsmzdvLrjCLqNjx4507NixSF7rcqpVq4bNZsu+BAcH07p1axYsWFAkrz9//nxsNhsHDhzIHsvrezNp0iS+/PLLAqsty4EDB7DZbMyfPz9X99+/fz+PP/449evXJygoCLvdTrVq1bjnnntYuXIlhmFk3zenzz+3+vfvT3BwsMuP+6/nrFat2n/ezzAMFi1aRPv27YmIiMBut1O5cmW6du3Ku+++m6fXLi5fEyLiORSCRMStzZ07F4Bt27bxxx9/5Ok5oqOjGTduXJGFoOKkXbt2/P777/z+++/ZP5T369ePmTNnWlLPjBkzmDFjhsuPK6wQ5Iqvv/6axo0b8/XXX9OvXz8WL17MsmXLeP755zl58iSdO3fmp59+srTGgjBq1Cj69u1L/fr1effdd1myZAkTJkygfPnyfPXVV1aXJyKSKz5WFyAiklcbNmzgr7/+okePHnz33XfMmTOH1q1bW12WWylVqhRXXXVV9vUuXbpQtWpVXn/9dR555JEcH5ORkUF6ejr+/v4FXk+DBg0K/DmLwt69e+nbty8NGzbkhx9+IDQ0NPu2Dh068MADD7Bq1SpKly5tYZX5d/bsWd544w3uu+8+Zs2a5XRb//79yczMtKgyERHXaCZIRNzWnDlzAHjppZdo27YtixYt4syZMxfd78iRIzz88MNERUXh5+dHZGQkt912G8eOHWPVqlVceeWVAAwYMCB7adjYsWOBSy/TyWnp0Lhx42jdujVlypQhNDSUFi1aMGfOHKclULnVu3dvqlatmuMPla1bt6ZFixbZ1z/99FNat25NWFgYgYGB1KhRg/vvv9/l1wQzFNWtW5eDBw8CjuVgU6ZMYcKECVSvXh1/f39WrlwJmEG0Z8+elClTBrvdTvPmzfnkk08uet61a9fSrl077HY7kZGRjBo1inPnzl10v5ze79TUVMaPH0/9+vWx2+2Eh4fTqVMnfvvtNwBsNhvJycm899572X9/5z/H0aNHGThwIJUrV8bPz4/q1aszbtw40tPTnV4nOjqaO+64g5CQEMLCwujTpw9Hjx7N1fv2+uuvc+bMGWbMmOEUgC783Jo2bfqfzzV37lyaNm2K3W6nTJky3HzzzezYsSPH+27bto1rr72WoKAgypUrx2OPPXbR18Bbb73FNddcQ0REBEFBQTRu3JgpU6bk+P7/l+TkZFJTU6lYsWKOt3t5Of9YkZ+vibS0NCZMmEC9evXw9/enXLlyDBgwgOPHjzvd76effqJjx46Eh4cTEBBAlSpVuPXWW3P8XiAikkUzQSLils6ePcvChQu58soradSoEffffz8PPvggn376Kf369cu+35EjR7jyyis5d+4co0ePpkmTJpw8eZJly5Zx+vRpWrRowbx58xgwYADPPfccPXr0AKBy5cou13TgwAEGDhxIlSpVAPMH/yFDhnDkyBFeeOEFl57r/vvvp1evXvz000906dIle/yff/5h3bp1TJ8+HYDff/+dPn360KdPH8aOHYvdbufgwYN5XnZ17tw5Dh48SLly5ZzGp0+fTp06dXj11VcJDQ2ldu3arFy5khtuuIHWrVvz9ttvExYWxqJFi+jTpw9nzpyhf//+AGzfvp1rr72WatWqMX/+fAIDA5kxYwYfffTRf9aTnp5Ot27d+PXXXxk2bBidO3cmPT2dtWvXcujQIdq2bcvvv/9O586d6dSpE88//zxAdhA5evQorVq1wsvLixdeeIGaNWvy+++/M2HCBA4cOMC8efMA899Tly5diI6OZvLkydSpU4fvvvuOPn365Op9W7FiBRUrVqRly5a5fatzNHnyZEaPHk3fvn2ZPHkyJ0+eZOzYsbRp04b169dTu3bt7PueO3eO7t27M3DgQJ555hl+++03JkyYwMGDB/nmm2+y77d3717uuusuqlevjp+fH3/99RcTJ07kn3/+yV5Omltly5al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TsPHxye7G9nhw4eN+++/36hQoYLh6+trREZGGnfccYdx7Nix7McsXLjQqFevnuHr62sAxpgxY7Jve++994z69esbdrvdaNCggfHxxx/n2Elr7ty5Rt26dQ1/f3+jRo0axuTJk405c+b8Zwe0/3LXXXcZgNGuXbuLbvv222+Nbt26GZUqVTL8/PyMiIgIo3v37savv/76n89btWpVo0ePHpe9T1Z3uFdeeSXH2//66y/jjjvuMCIiIgxfX1+jQoUKRufOnbO79mVZs2aNcdVVVxn+/v5GhQoVjKeeesqYNWtWrt6bs2fPGi+88IJRu3Ztw8/PzwgPDzc6d+5s/Pbbb9n32bx5s9GuXTsjMDDQAJye4/jx48bjjz9uVK9e3fD19TXKlCljXHHFFcazzz5rJCUlZd/v33//NW699VYjODjYCAkJMW699Vbjt99+y1V3uCx79+41hgwZYtStW9cICAgw/P39japVqxq33367sXjxYqeOaBd2h8vy7rvvGk2aNDH8/PyMsLAwo1evXtld7LL069fPCAoKMv7++2+jY8eORkBAgFGmTBnjkUcecfqcDMMwvvnmG6Np06aG3W43KlWqZDz11FPGkiVLDMBYuXKl03P+V3e41NRU49VXXzW6detmVKlSxfD39zfsdrtRv359Y+TIkcbJkyed7p+fr4lz584Zr776anbtwcHBRr169YyBAwcau3fvNgzDMH7//Xfj5ptvNqpWrWr4+/sb4eHhRocOHYyvv/76sp+HiIjNMPJwip+IiIiIiIib0p4gERERERHxKApBIiIiIiLiURSCRERERETEoygEiYiIiIiIR1EIEhERERERj6IQJCIiIiIiHsWtD0vNzMwkOjqakJCQ7FOxRURERETE8xiGQWJiIpGRkXh5XX6ux61DUHR0NFFRUVaXISIiIiIixcThw4epXLnyZe/j1iEoJCQEMD/R0NBQi6sRERERERGrJCQkEBUVlZ0RLsetQ1DWErjQ0FCFIBERERERydU2GTVGEBERERERj6IQJCIiIiIiHkUhSEREREREPIpCkIiIiIiIeBSFIBERERER8SgKQSIiIiIi4lEUgkRERERExKMoBImIiIiIiEdRCBIREREREY+iECQiIiIiIh5FIUhERERERDyKQpCIiIiIiHgUhSAREREREfEoCkEiIiIiIuJRFIJERERERMSjKASJiIiIiIhHUQgSERERERHXHTkC27dbXUWeKASJiIiIiEjunT0LEydCnTrQrx9kZlpdkcsUgkREREREJHeWLoUGDeC55+DMGdiwAd5/3+qqXOZjdQEiIiIiIuImzp6FAwfMj7294ZFH4KabLC0pLxSCREREREQkd3r3hs6dwWaDN96ARo2srihPFIJERERERMTZuXPw9tvwxx/wwQeOcZsNvvwSgoPNj92UQpCIiIiIiDj88AMMHero/HbXXdC9u+P2kBBr6ipAaowgIiIiIiKwd6+53O2665xbX69da1lJhUUzQSIiIiIiniwxESZNgtdfh7Q0x3irVjB9OrRubV1thUQhSERERETEE2Vmmvt9nnkGYmIc4xUrwssvw913g1fJXDimECQiIiIi4ol++8087DSLnx888QSMGlUi9v1cTsmMdiIiIiIicnlXX+0446d3b9ixw1wWV8IDECgEiYiIiIiUfCkpsGABGIbz+Ouvw4oVsHgx1KhhTW0W0HI4EREREZGSyjDMc32eeAL274fAQLjtNsfttWqZFw+jmSARERERkZJo61az3fUtt5gBCODppyEjw9q6igGFIBERERGRkuTUKRgyBJo1gx9/dIx36mTOCnl7W1VZsaHlcCIiIiIiJUF6OsyaBc8/bwahLNWqwWuvwc03g81mWXnFiUKQiIiIiIi7S06GNm1gyxbHWGAgPPssjBgBdrt1tRVDWg4nIiIiIuLugoKgQQPH9XvugV27YPRoBaAcaCZIRERERMTdJCdDQAB4nTenMWUKxMTASy+Zs0JySZoJEhERERFxF5mZ8MEHUKeO+ef5qlSBn39WAMoFhSAREREREXewbh20awf33gvR0Wa768REq6tySwpBIiIiIiLFWUwMDBgArVvD2rWO8VatICnJurrcmEKQiIiIiEhxlJpq7vOpUwfmz3eM168PS5fCV19BxYqWlefO1BhBRERERKS4+eYbGD4c9u51jIWFwbhx8Oij4OtrXW0lgKUzQdWqVcNms110GTx4sJVliYiIiIhY68svHQHIywsGDYLdu2HoUAWgAmBpCFq/fj0xMTHZlxUrVgBw++23W1mWiIiIiIi1Jk6EkBDo0AE2boSZM6FcOaurKjEsXQ5X7oK/yJdeeomaNWvSoUMHiyoSERERESlCGRkwe7Z52Om99zrGK1Qww0+tWmCzWVdfCVVs9gSlpaXxwQcfMGLECGyX+ItOTU0lNTU1+3pCQkJRlSciIiIiUrBWrTKXt/39N4SHw403QunSjttr17astJKu2HSH+/LLL4mLi6N///6XvM/kyZMJCwvLvkRFRRVdgSIiIiIiBeHgQbj9dujUyQxAACdPwtdfW1uXB7EZhmFYXQRA165d8fPz45tvvrnkfXKaCYqKiiI+Pp7Q0NCiKFNEREREJG+Sk+Hll+GVVyAlxTHeogVMmwZXX21dbSVAQkICYWFhucoGxWI53MGDB/nhhx/44osvLns/f39//P39i6gqEREREZECYBiwaBGMHAn//usYj4iASZOgf3/w9rasPE9ULELQvHnziIiIoEePHlaXIiIiIiJSsF5/HZ580nHd19fcC/Tcc+bZP1LkLN8TlJmZybx58+jXrx8+PsUik4mIiIiIFJwBA6BMGfPjHj1g61ZzSZwCkGUsTx0//PADhw4d4v7777e6FBERERGR/ElLM5sdtGzpGCtTBmbMgNBQ6NbNutokW7FpjJAXrmx+EhEREREpNIYB338Pw4fDsWOwaxeUL291VR7FlWxg+XI4ERERERG39s8/0L27ec7P7t2QkADPPmt1VXIZCkEiIiIiInkRF2fO/DRuDEuXOsavvhoefdSysuS/Wb4nSERERETErWRkwJw55mzPiROO8agos+HBHXeAzWZdffKfFIJERERERHJr924z5Gze7Biz2+Hpp81zgAIDLStNck8hSEREREQktypUgKNHHdf79IEpU6BKFetqEpdpT5CIiIiIyKVc2Eg5JAQmT4ZmzeCXX2DRIgUgN6QQJCIiIiJyIcOAjz+GBg3g8GHn2+67DzZsgPbtralN8k0hSERERETkfH/+CR06wJ13mu2vn37a+XYvL/D2tqY2KRAKQSIiIiIiALGx8PDDcMUV8OuvjvG4OEhLs6wsKXgKQSIiIiLi2dLSYOpUqFMHZs927AOqXRu+/Ra+/x78/KytUQqUusOJiIiIiOdauhSGDYOdOx1jISEwZgwMGaLwU0IpBImIiIiIZ0pNhYED4dAh87rNBvffDxMnQvny1tYmhUrL4URERETEM/n7w6uvmh+3awfr18O77yoAeQCFIBEREREp+TIyYM4c2LPHefy222DJErMRwhVXWFObFDmFIBEREREp2Vavhlat4MEH4YknnG+z2eCGG8w/xWMoBImIiIhIyXT4MNx1l3mo6aZN5tjXX8PmzZaWJdZTCBIRERGRkuXsWXjxRahXDxYudIw3aQIrV0KzZpaVJsWDusOJiIiISMlgGPD55/Dkk3DwoGM8PBwmTDCXw/nox19RCBIRERGRkuKhh8zmB1m8veGxx8wzf0qXtq4uKXa0HE5ERERESoabb3Z8fN118Pff8MYbCkByEc0EiYiIiIj7OXcOTp+GiAjHWPfu8Mgj0K0b3HijOr7JJSkEiYiIiIh7Wb4chg2DyEhYscIRdmw2mDHD0tLEPWg5nIiIiIi4h927oWdP6NoVduyAH3+Er76yuipxQwpBIiIiIlK8JSTA009Dw4bwzTeO8auugipVrKtL3JaWw4mIiIhI8ZSZCe+9B6NGwbFjjvHISJgyxTwIVft+JA8UgkRERESk+Nm4EQYNgg0bHGP+/uYZQM88A8HB1tUmbk8hSERERESKn4QE5wB0663wyitQvbp1NUmJoRAkIiIiIsVPp05m8Nm1yzzrp3NnqyuSEkQhSERERESsYxiweDF8/jl88IHzHp/ZsyEkBHz0I6sULHWHExERERFrbNkCXbqYMz4ffQQLFzrfXrq0ApAUCoUgERERESlaJ0/C4MHQrBn89JNjfOlSy0oSz6IQJCIiIiJFIz0d3nwTateGGTPMFtgANWqYS+Lee8/a+sRjaH5RRERERArfDz/AsGGwbZtjLCgInnvOHLfbrapMPJBCkIiIiIgUrp074brrnMfuuw8mTzYPPhUpYloOJyIiIiKFq25duPde8+NWreD3382lbwpAYhGFIBEREREpOJmZ8MUXkJHhPP7SSzB/vhmArrrKktJEsigEiYiIiEjB+OMPaNvWbHk9d67zbZGR0K8feOnHT7Ge/hWKiIiISP5ER5sB56qrzCAE8OyzkJxsbV0il6AQJCIiIiJ5k5JiNjeoUwcWLHCMN2xoHn4aFGRdbSKXoe5wIiIiIuIaw4CvvoInnoB9+xzjpUvD+PEwaBD46MdMKb70r1NEREREci8jA3r0gGXLHGNeXjBwoBmAypa1rjaRXFIIEhEREZHc8/aG6tUd1zt2hGnToEkTy0oScZX2BImIiIjIpaWnm5fzvfgiNG8On30GP/2kACRuRyFIRERERHK2ciW0aAFvveU8XrYsbNxotsK22aypTSQfLA9BR44c4Z577iE8PJzAwECaNWvGxo0brS5LRERExHPt328GnM6dYcsWGDMGjh93vo/Cj7gxS/cEnT59mnbt2tGpUyeWLFlCREQEe/fupVSpUlaWJSIiIuKZkpLgpZfg1VchNdUxXrs2nDoF5cpZV5tIAbI0BL388stERUUxb9687LFq1apZV5CIiIiIJzIM+PBDePpp8+DTLOXLm6HovvvMDnAiJYSl/5q//vprWrZsye23305ERATNmzdn9uzZl7x/amoqCQkJThcRERERyYf166FdO7j3XkcA8vWFkSNh1y7o318BSEocS/9F79u3j5kzZ1K7dm2WLVvGoEGDePzxx1lw/onD55k8eTJhYWHZl6ioqCKuWERERKSE+fhj+P13x/WbboJt2+DllyE01Lq6RAqRzTAMw6oX9/Pzo2XLlvz222/ZY48//jjr16/n9/O/GP9famoqqeetT01ISCAqKor4+HhC9UUqIiIi4rr4eKhTB8LDYepU6NrV6opE8iQhIYGwsLBcZQNL9wRVrFiRBg0aOI3Vr1+fzz//PMf7+/v74+/vXxSliYiIiJQshgHffgvHjsGDDzrGw8LMs37q1DGXwYl4AEuXw7Vr146dO3c6je3atYuqVataVJGIiIhICbR9O9xwA/TsCcOGwZEjzrc3bKgAJB7F0hA0fPhw1q5dy6RJk9izZw8fffQRs2bNYvDgwVaWJSIiIlIynD5thp4mTWD5cnMsORnmz7eyKhHLWbonCODbb79l1KhR7N69m+rVqzNixAgeeuihXD3WlXV/IiIiIh4jIwPefReeew5OnHCMV6lingF022067FRKHFeygeUhKD8UgkREREQu8PPPMHQo/PWXYywgAJ55Bp58EgIDratNpBC5TWMEERERESlAixZB377OY337mu2udbSISDadfCUiIiJSUtx0E1SubH7cvDn8+it89JECkMgFNBMkIiIi4o4MA7ZuhcaNHWNBQfC//5n7gAYMAG9v6+oTKcYUgkRERETczcaN5r6fDRvM9tc1ajhu693bsrJE3IWWw4mIiIi4i6yDTq+8EtasgdRUeOopq6sScTuaCRIREREp7tLSzGVu48dDQoJjvG5dMxSJiEsUgkRERESKs+++g+HDYfdux1hoKIwdC4MHg5+fZaWJuCuFIBEREZHi6Ngxs7nBkiWOMZvNnPmZMAEiIqyrTcTNKQSJiIiIFEel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Wbt2LQD79+/HjRvNiYiIeIZ//oEePWDdOufxa66BQ4dgxAjzDCARkRLM5RDUuXNnvvnmGwAeeOABhg8fznXXXUefPn24+eabC7xAERERKQBxcfDEE+Y5P99/bzZAyMx0vo+OmxARD+HyOUGZmZlkZmbi42M2lvvkk09YvXo1tWrVYtCgQfgV4W+PdE6QiIjIf8jIgLlz4dln4fhxx3jlyrB6NVStal1tIiIFyJVsoMNSRURESqpffzXbW//5p2PMboeRI+HppyEw0LraREQKmCvZINfL4U6dOsW/5x+aBmzbto0BAwZwxx138NFHH+WtWhERESlYhw7BnXea+3zOD0B33GHuCRo3TgFIRDxarkPQ4MGDef3117Ovx8bG0r59e9avX09qair9+/fn/fffL5QiRURExAUDBsDHHzuuN20Kq1aZY1r+JiKS+xC0du1aevbsmX19wYIFlClThs2bN/PVV18xadIk3nrrrUIpUkRERFwwebL5Z3g4vP02bNwIHTpYW5OISDGS6xB09OhRqlevnn39p59+4uabb85ukNCzZ092795d8BWKiIjIpW3eDBs2OI+1agUffgi7d8PAgeY5QCIiki3XISg0NJS4uLjs6+vWreOqq67Kvm6z2UhNTS3Q4kREROQSjh83A06LFvDgg2YXuPPddReULm1NbSIixVyuQ1CrVq2YPn06mZmZfPbZZyQmJtK5c+fs23ft2kVUVFShFCkiIiL/79w5eOMNqF0bZs0Cw4C//jJnfkREJFd8cnvHF198kS5duvDBBx+Qnp7O6NGjKX3eb5gWLVpEB603FhERKTxLl8Lw4WaHtywhIfDCC2Y3OBERyZVch6BmzZqxY8cOfvvtNypUqEDr1q2dbr/zzjtp0KBBgRcoIiLi8XbtghEj4LvvHGM2m9kFbtIkKF/eutpERNyQDksVEREpzubMgUceMZfBZWnbFqZNg5YtratLRKSYKZTDUkVERMQCrVo5mh5UqmTu/Vm9WgFIRCQfcr0cTkRERIpAUhIEBzuuN25s7gMKCIBnnoGgIOtqExEpIRSCREREioN//4Wnn4ZNm8xub35+jttefdW6ukRESiAthxMREbHS2bPw4otQty589JHZ+e2tt6yuSkSkRMvVTFBCQkKun1ANCkRERHLBMODzz+HJJ+HgQcd4eLgOORURKWS5CkGlSpXCZrNd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kVaX4jE0E5RLZcuWxdvb+6JZn9jY2Itmh0RKmiFDhvD111+zcuVKKleubHU5HsHPz49atWrRsmVLJk+eTNOmTZk2bZrVZZV4GzduJDY2liuuuAIfHx98fHz4+eefmT59Oj4+PmRkZFhdoscICgqicePG7N692+pSSrSKFSte9MuV+vXrq+lTETp48CA//PADDz74oNWleBSFoFzy8/PjiiuuyO7ckWXFihW0bdvWoqpECpdhGDz22GN88cUX/PTTT1SvXt3qkjyWYRikpqZaXUaJd+2117JlyxY2b96cfWnZsiV33303mzdvxtvb2+oSPUZqaio7duygYsWKVpdSorVr1+6iow927dpF1apVLarI88ybN4+IiAh69OhhdSkeRcvhXDBixAjuvfdeWrZsSZs2bZg1axaHDh1i0KBBVpdW4iUlJbFnz57s6/v372fz5s2UKVOGKlWqWFhZyTZ48GA++ugjvvrqK0JCQrJnQsPCwggICLC4upJr9OjRdOvWjaioKBITE1m0aBGrVq1i6dKlVpdW4oWEhFy05y0oKIjw8HDthStkTz75JDfddBNVqlQhNjaWCRMmkJCQQL9+/awurUQbPnw4bdu2ZdKkSdxxxx2sW7eOWbNmMWvWLKtL8wiZmZnMmzePfv364eOjH8uLkt5tF/Tp04eTJ08yfvx4YmJiaNSoEd9//71+W1IENmzYQKdOnbKvZ60P79evH/Pnz7eoqpIvqx18x44dncbnzZunzZuF6NixY9x7773ExMQQFhZGkyZNWLp0Kdddd53VpYkUmn///Ze+ffty4sQJypUrx1VXXcXatWv1f2whu/LKK1m8eDGjRo1i/PjxVK9enTfeeIO7777b6tI8wg8//MChQ4e4//77rS7F4+icIBERERER8SjaEyQiIiIiIh5FIUhERERERDyKQpCIiIiIiHgUhSAREREREfEoCkEiIiIiIuJRFIJERERERMSjKASJiIiIiIhHUQgSERERERGPohAkIiKFymaz8eWXX1ryXP3796d37975es0DBw5gs9nYvHlzvp7nv4wdO5ZmzZoV6muIiIhJIUhEpIT47bff8Pb25oYbbnD5sdWqVeONN94o+KJy6ejRowwdOpRatWpht9spX748V199NW+//TZnzpyxrK7c2rdvH3379iUyMhK73U7lypXp1asXu3btsro0ERHJgY/VBYiISMGYO3cuQ4YM4d133+XQoUNUqVLF6pJyZd++fbRr145SpUoxadIkGjduTHp6Ort27WLu3LlERkbSs2dPq8u8pLS0NK677jrq1avHF198QcWKFfn333/5/vvviY+Pt7o8ERHJgWaCRERKgOTkZD755BMeeeQRbrzxRubPn3/Rfb7++mtatmyJ3W6nbNmy3HLLLQB07NiRgwcPMnz4cGw2GzabDch5edYbb7xBtWrVsq+vX7+e6667jrJlyxIWFkaHDh3YtGmTS7U/+uij+Pj4sGHDBu644w7q169P48aNufXWW/nuu++46aabLvnYLVu20LlzZwICAggPD+fhhx8mKSnpovuNGzeOiIgIQkNDGThwIGlpadm3LV26lKuvvppSpUoRHh7OjTfeyN69e3Nd//bt29m3bx8zZszgqquuomrVqrRr146JEydy5ZVXZt/v6aefpk6dOgQGBlKjRg2ef/55zp07d9nnnjdvHvXr18dut1OvXj1mzJiRfVtaWhqPPfYYFStWxG63U61aNSZPnpzrukVEPJlCkIhICfDxxx9Tt25d6tatyz333MO8efMwDCP79u+++45bbrmFHj168Oeff/Ljjz/SsmVLAL744gsqV67M+PHjiYmJISYmJtevm5iYSL9+/fj1119Zu3YttWvXpnv37iQmJubq8SdPnmT58uUMHjyYoKCgHO+TFcoudObMGW644QZKly7N+vXr+fTTT/nhhx947LHHnO73448/smPHDlauXMnChQtZvHgx48aNy749OTmZESNGsH79en788Ue8vLy4+eabyczMzNXnUK5cOby8vPjss8/IyMi45P1CQkKYP38+27dvZ9q0acyePZupU6de8v6zZ8/m2WefZeLEiezYsYNJkybx/PPP89577wEwffp0vv76az755BN27tzJBx984BRQRUTkMgwREXF7bdu2Nd544w3DMAzj3LlzRtmyZY0VK1Zk396mTRvj7rvvvuTjq1atakydOtVpbMyYMUbTpk2dxqZOnWpUrVr1ks+Tnp5uhISEGN988032GGAsXrw4x/uvXbvWAIwvvvjCaTw8PNwICgoygoKCjJEjR+b4XLNmzTJKly5tJCUlZd/+3XffGV5eXsbRo0cNwzCMfv36GWXKlDGSk5Oz7zNz5kwjODjYyMjIyLGm2NhYAzC2bNliGIZh7N+/3wCMP//885Kf95tvvmkEBgYaISEhRqdOnYzx48cbe/fuveT9DcMwpkyZYlxxxRXZ1y98v6OiooyPPvrI6TEvvvii0aZNG8MwDGPIkCFG586djczMzMu+joiIXEwzQSIibm7nzp2sW7eOO++8EwAfHx/69OnD3Llzs++zefNmrr322gJ/7djYWAYNGkSdOnUICwsjLCyMpKQkDh065NLzXDjbs27dOjZv3kzDhg1JTU3N8TE7duygadOmTjNI7dq1IzMzk507d2aPNW3alMDAwOzrbdq0ISkpicOHDwOwd+9e7rrrLmrUqEFoaCjVq1cHcOlzGDx4MEePHuWDDz6gTZs2fPrppzRs2JAVK1Zk3+ezzz7j6quvpkKFCgQHB/P8889f8jWOHz/O4cOHeeCBBwgODs6+TJgwIXupXv/+/dm8eTN169bl8ccfZ/ny5bmuV0TE06kxgoiIm5szZw7p6elUqlQpe8wwDHx9fTl9+jSlS5cmICDA5ef18vJyWlIHXLSHpX///hw/fpw33niDqlWr4u/vT5s2bZz23FxOrVq1sNls/PPPP07jNWrUALhs3YZhXHKp3KXGc7rPTTfdRFRUFLNnzyYyMpLMzEwaNWqU688hS0hICD179qRnz55MmDCBrl27MmHCBK677jrWrl3LnXfeybhx4+jatSthYWEsWrSI1157LcfnylqKN3v2bFq3bu10m7e3NwAtWrRg//79LFmyhB9++IE77riDLl268Nlnn7lUt4iIJ9JMkIiIG0tPT2fBggW89tprbN68Ofvy119/UbVqVT788EMAmjRpwo8//njJ5/Hz87toP0u5cuU4evSoUxC68KycX3/9lccff5zu3bvTsGFD/P39OXHiRK7rDw8P57rrruPNN98kOTk5148DaNCgAZs3b3Z63Jo1a/Dy8qJOnTrZY3/99Rdnz57Nvr527VqCg4OpXLkyJ0+eZMeOHTz33HNce+211K9fn9OnT7tUR05sNhv16tXLrm3NmjVUrVqVZ599lpYtW1K7dm0OHjx4yceXL1+eSpUqsW/fPmrVquV0yZqpAggNDaVPnz7Mnj2bjz/+mM8//5xTp07lu34RkZJOIUhExI19++23nD59mgceeIBGjRo5XW677TbmzJkDwJgxY1i4cCFjxoxhx44dbNmyhSlTpmQ/T7Vq1fjll184cuRIdojp2LEjx48fZ8qUKezdu5e33nqLJUuWOL1+rVq1eP/999mxYwd//PEHd999t8uzTjNmzCA9PZ2WLVvy8ccfs2PHjuyN/v/880/2zMeF7r77bux2O/369WPr1q2sXLmSIUOGcO+991K+fPns+6WlpfHAAw+wfft2lixZwpgxY3jsscfw8vKidOnShIeHM2vWLPbs2cNPP/3EiBEjXKp/8+bN9OrVi88++4zt27ezZ88e5syZw9y5c+nVq1f2+3To0CEWLVrE3r17mT59OosXL77s844dO5bJkyczbdo0du3axZYtW5g3bx6vv/46AFOnTmXRokX8888/7Nq1i08//ZQKFSpQqlQpl+oXEfFI1m5JEhGR/LjxxhuN7t2753jbxo0bDcDYuHGjYRiG8fnnnxvNmjUz/Pz8jLJlyxq33HJL9n1///13o0mTJoa/v79x/n8NM2fONKKiooygoCDjvvvuMyZOnOjUGGHTpk1Gy5YtDX9/f6N27drGp59+elGTBS7TGCFLdHS08dhjjxnVq1c3fH19jeDgYKNVq1bGK6+84tTU4MLn+vvvv41OnToZdrvdKFOmjPHQQw8ZiYmJ2bf369fP6NWrl/HCCy8Y4eHhRnBwsPHggw8aKSkp2fdZsWKFUb9+fcPf399o0qSJsWrVKqfX+a/GCMePHzcef/xxo1GjRkZwcLAREhJiNG7c2Hj11Vedmi889dRT2TX06dPHmDp1qhEWFpZ9e06NKD788MPsv7PSpUsb11xzTXYTiVmzZhnNmjUzgoKCjNDQUOPaa681Nm3adNn3WURETDbDuGDBt4iIiIiISAmm5XAiIiIiIuJRFIJERERERMSjKASJiIiIiIhHUQgSERERERGPohAkIiIiIiIeRSFIREREREQ8ikKQiIiIiIh4FIUgERERERHxKApBIiIiIiLiURSCRERERETEoygEiYiIiIiIR/k/u+VB++AbZq0AAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 1000x600 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Prepare features (X) and target variable (y) for the prediction model\n",
    "X = df_cleaned[['Critic_Score', 'User_Score']]\n",
    "y = df_cleaned['Global_Sales']\n",
    "\n",
    "# Split the data into training and testing sets\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "\n",
    "# Create and train a linear regression model\n",
    "model = LinearRegression()\n",
    "model.fit(X_train, y_train)\n",
    "\n",
    "# Make predictions on the test set\n",
    "y_pred = model.predict(X_test)\n",
    "\n",
    "# Calculate model performance metrics\n",
    "mse = mean_squared_error(y_test, y_pred)\n",
    "r2 = r2_score(y_test, y_pred)\n",
    "\n",
    "print(f\"Mean Squared Error: {mse:.2f}\")\n",
    "print(f\"R-squared: {r2:.2f}\")\n",
    "\n",
    "# Visualize actual vs predicted sales\n",
    "plt.figure(figsize=(10, 6))\n",
    "plt.scatter(y_test, y_pred, alpha=0.5)\n",
    "plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)\n",
    "plt.xlabel('Actual Global Sales')\n",
    "plt.ylabel('Predicted Global Sales')\n",
    "plt.title('Actual vs Predicted Global Sales')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bfe82e30-a3d4-4f18-ace4-99339842c2c5",
   "metadata": {},
   "source": [
    "##### Step 3: Interactive Visualization (Bonus)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "4b294fe8-a2cf-48c3-bcd9-cc0bd26725ff",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.plotly.v1+json": {
       "config": {
        "plotlyServerURL": "https://plot.ly"
       },
       "data": [
        {
         "customdata": [
          [
           "Call of Duty: Modern Warfare 2",
           "X360",
           2009
          ],
          [
           "Call of Duty 4: Modern Warfare",
           "X360",
           2007
          ],
          [
           "Call of Duty: World at War",
           "X360",
           2008
          ],
          [
           "Call of Duty 4: Modern Warfare",
           "PS3",
           2007
          ],
          [
           "Resistance: Fall of Man",
           "PS3",
           2006
          ],
          [
           "Left 4 Dead",
           "X360",
           2008
          ],
          [
           "Battlefield: Bad Company 2",
           "X360",
           2010
          ],
          [
           "Killzone 2",
           "PS3",
           2009
          ],
          [
           "Battlefield: Bad Company 2",
           "PS3",
           2010
          ],
          [
           "BioShock",
           "X360",
           2007
          ],
          [
           "Call of Duty 3",
           "X360",
           2006
          ],
          [
           "Resistance 2",
           "PS3",
           2008
          ],
          [
           "Call of Duty 3",
           "Wii",
           2006
          ],
          [
           "Call of Duty 2",
           "X360",
           2005
          ],
          [
           "Call of Duty: World at War",
           "Wii",
           2008
          ],
          [
           "Crackdown",
           "X360",
           2007
          ],
          [
           "Call of Duty: Modern Warfare: Reflex Edition",
           "Wii",
           2009
          ],
          [
           "The House of the Dead 2 & 3 Return",
           "Wii",
           2008
          ],
          [
           "BioShock",
           "PS3",
           2008
          ],
          [
           "MAG: Massive Action Game",
           "PS3",
           2010
          ],
          [
           "Army of Two",
           "PS3",
           2008
          ],
          [
           "Star Fox: Assault",
           "GC",
           2005
          ],
          [
           "Resistance: Retribution",
           "PSP",
           2009
          ],
          [
           "Perfect Dark Zero",
           "X360",
           2005
          ],
          [
           "Red Steel 2",
           "Wii",
           2010
          ],
          [
           "Metroid Prime: Trilogy",
           "Wii",
           2009
          ],
          [
           "Doom (2016)",
           "NS",
           2017
          ],
          [
           "F.E.A.R. 2: Project Origin",
           "PC",
           2009
          ],
          [
           "Wolfenstein",
           "PC",
           2009
          ],
          [
           "Left 4 Dead",
           "PC",
           2008
          ]
         ],
         "hovertemplate": "Genre=Shooter<br>Critic_Score=%{x}<br>Global_Sales=%{y}<br>Name=%{customdata[0]}<br>Platform=%{customdata[1]}<br>Year=%{customdata[2]}<extra></extra>",
         "legendgroup": "Shooter",
         "marker": {
          "color": "#636efa",
          "symbol": "circle"
         },
         "mode": "markers",
         "name": "Shooter",
         "orientation": "v",
         "showlegend": true,
         "type": "scatter",
         "x": [
          9.5,
          9.6,
          8.5,
          9.5,
          8.5,
          8.7,
          9,
          9.1,
          8.9,
          9.6,
          8,
          8.5,
          6.8,
          8.8,
          8.1,
          8.4,
          7.2,
          6.5,
          9.3,
          7.4,
          7.4,
          6.6,
          8,
          8.1,
          7.8,
          9,
          8.2,
          8.2,
          6.5,
          8.6
         ],
         "xaxis": "x",
         "y": [
          13.53,
          9.41,
          7.5,
          6.72,
          4.37,
          3.52,
          3.48,
          3.02,
          2.96,
          2.83,
          2.7,
          2.47,
          2.24,
          2.06,
          1.94,
          1.75,
          1.51,
          1.45,
          1.44,
          1.32,
          1.17,
          1.08,
          0.9,
          0.77,
          0.62,
          0.61,
          0.43,
          0.09,
          0.04,
          0.02
         ],
         "yaxis": "y"
        },
        {
         "customdata": [
          [
           "Grand Theft Auto IV",
           "PS3",
           2008
          ],
          [
           "Grand Theft Auto V",
           "XOne",
           2014
          ],
          [
           "Uncharted 3: Drake's Deception",
           "PS3",
           2011
          ],
          [
           "Uncharted 2: Among Thieves",
           "PS3",
           2009
          ],
          [
           "Red Dead Redemption",
           "X360",
           2010
          ],
          [
           "Metal Gear Solid 2: Sons of Liberty",
           "PS2",
           2001
          ],
          [
           "Metal Gear Solid 4: Guns of the Patriots",
           "PS3",
           2008
          ],
          [
           "Assassin's Creed",
           "X360",
           2007
          ],
          [
           "Resident Evil 5",
           "PS3",
           2009
          ],
          [
           "Uncharted: Drake's Fortune",
           "PS3",
           2007
          ],
          [
           "Assassin's Creed",
           "PS3",
           2007
          ],
          [
           "Monster Hunter: World",
           "PS4",
           2018
          ],
          [
           "Tom Clancy's Splinter Cell",
           "XB",
           2002
          ],
          [
           "inFAMOUS",
           "PS3",
           2009
          ],
          [
           "Star Wars: The Force Unleashed",
           "Wii",
           2008
          ],
          [
           "Devil May Cry 4",
           "PS3",
           2008
          ],
          [
           "Tom Clancy's Splinter Cell: Pandora Tomorrow",
           "XB",
           2004
          ],
          [
           "Resident Evil 4: Wii Edition",
           "Wii",
           2007
          ],
          [
           "Bayonetta",
           "PS3",
           2010
          ],
          [
           "Dante's Inferno",
           "PS3",
           2010
          ],
          [
           "Bayonetta",
           "X360",
           2010
          ],
          [
           "MadWorld",
           "Wii",
           2009
          ],
          [
           "Tom Clancy's Splinter Cell: Double Agent",
           "X360",
           2006
          ],
          [
           "Harry Potter and the Half-Blood Prince",
           "Wii",
           2009
          ],
          [
           "X-Men Origins: Wolverine - Uncaged Edition",
           "PS3",
           2009
          ],
          [
           "Tom Clancy's HAWX",
           "PS3",
           2009
          ],
          [
           "No More Heroes",
           "Wii",
           2008
          ],
          [
           "Fire Emblem Warriors",
           "NS",
           2017
          ],
          [
           "Golden Axe: Beast Rider",
           "X360",
           2008
          ],
          [
           "Dead Rising: Chop Till You Drop",
           "Wii",
           2009
          ],
          [
           "Darksiders",
           "PS3",
           2010
          ],
          [
           "Deadly Premonition",
           "X360",
           2010
          ],
          [
           "Deadly Creatures",
           "Wii",
           2009
          ],
          [
           "Dragon Ball: Revenge of King Piccolo",
           "Wii",
           2009
          ],
          [
           "Disaster: Day of Crisis",
           "Wii",
           2008
          ],
          [
           "Devil's Third",
           "WiiU",
           2015
          ]
         ],
         "hovertemplate": "Genre=Action<br>Critic_Score=%{x}<br>Global_Sales=%{y}<br>Name=%{customdata[0]}<br>Platform=%{customdata[1]}<br>Year=%{customdata[2]}<extra></extra>",
         "legendgroup": "Action",
         "marker": {
          "color": "#EF553B",
          "symbol": "circle"
         },
         "mode": "markers",
         "name": "Action",
         "orientation": "v",
         "showlegend": true,
         "type": "scatter",
         "x": [
          10,
          9,
          9.3,
          9.5,
          9.5,
          9.5,
          9.3,
          8.2,
          8.6,
          8.7,
          8.2,
          9.3,
          9.3,
          8.7,
          6.8,
          8.2,
          9.2,
          9.2,
          8.6,
          7.3,
          8.9,
          8.1,
          8.5,
          6,
          7,
          7.5,
          8.2,
          7.3,
          4,
          6.7,
          7.8,
          6,
          6.9,
          6.1,
          6.4,
          3.8
         ],
         "xaxis": "x",
         "y": [
          10.57,
          8.72,
          6.84,
          6.74,
          6.5,
          6.05,
          6,
          5.55,
          5.1,
          4.97,
          4.83,
          4.67,
          3.02,
          2.99,
          1.86,
          1.58,
          1.48,
          1.46,
          1.21,
          1.08,
          0.93,
          0.78,
          0.78,
          0.76,
          0.74,
          0.56,
          0.56,
          0.51,
          0.33,
          0.29,
          0.28,
          0.26,
          0.22,
          0.2,
          0.07,
          0.05
         ],
         "yaxis": "y"
        },
        {
         "customdata": [
          [
           "LittleBigPlanet",
           "PS3",
           2008
          ],
          [
           "Jak and Daxter: The Precursor Legacy",
           "PS2",
           2001
          ],
          [
           "Ratchet & Clank Future: A Crack in Time",
           "PS3",
           2009
          ],
          [
           "Castlevania: Symphony of the Night",
           "PS",
           1997
          ],
          [
           "Mirror's Edge",
           "PS3",
           2008
          ],
          [
           "de Blob",
           "Wii",
           2008
          ],
          [
           "NiGHTS: Journey of Dreams",
           "Wii",
           2007
          ],
          [
           "Castlevania: Order of Ecclesia",
           "DS",
           2008
          ],
          [
           "Ratchet & Clank Future: Quest for Booty",
           "PSN",
           2008
          ],
          [
           "Wonder Boy: The Dragon's Trap (Remake)",
           "NS",
           2017
          ],
          [
           "Runner3",
           "NS",
           2018
          ]
         ],
         "hovertemplate": "Genre=Platform<br>Critic_Score=%{x}<br>Global_Sales=%{y}<br>Name=%{customdata[0]}<br>Platform=%{customdata[1]}<br>Year=%{customdata[2]}<extra></extra>",
         "legendgroup": "Platform",
         "marker": {
          "color": "#00cc96",
          "symbol": "circle"
         },
         "mode": "markers",
         "name": "Platform",
         "orientation": "v",
         "showlegend": true,
         "type": "scatter",
         "x": [
          9.4,
          9,
          8.7,
          9.3,
          7.7,
          8.1,
          7,
          8.2,
          7.7,
          7.4,
          7.5
         ],
         "xaxis": "x",
         "y": [
          5.85,
          3.64,
          1.89,
          1.27,
          1.13,
          0.96,
          0.41,
          0.37,
          0.09,
          0.05,
          0.02
         ],
         "yaxis": "y"
        },
        {
         "customdata": [
          [
           "Forza Motorsport 3",
           "X360",
           2009
          ],
          [
           "MotoGP '07",
           "X360",
           2007
          ]
         ],
         "hovertemplate": "Genre=Racing<br>Critic_Score=%{x}<br>Global_Sales=%{y}<br>Name=%{customdata[0]}<br>Platform=%{customdata[1]}<br>Year=%{customdata[2]}<extra></extra>",
         "legendgroup": "Racing",
         "marker": {
          "color": "#ab63fa",
          "symbol": "circle"
         },
         "mode": "markers",
         "name": "Racing",
         "orientation": "v",
         "showlegend": true,
         "type": "scatter",
         "x": [
          9.2,
          7.7
         ],
         "xaxis": "x",
         "y": [
          5.5,
          0.25
         ],
         "yaxis": "y"
        },
        {
         "customdata": [
          [
           "Final Fantasy XIII",
           "PS3",
           2010
          ],
          [
           "Fallout 3",
           "X360",
           2008
          ],
          [
           "Mass Effect 2",
           "X360",
           2010
          ],
          [
           "Mass Effect",
           "X360",
           2007
          ],
          [
           "The Legend of Dragoon",
           "PS",
           2000
          ],
          [
           "Demon's Souls",
           "PS3",
           2009
          ],
          [
           "The Elder Scrolls V: Skyrim",
           "NS",
           2017
          ],
          [
           "White Knight Chronicles: International Edition",
           "PS3",
           2010
          ],
          [
           "Lost Odyssey",
           "X360",
           2008
          ],
          [
           "Tales of Vesperia",
           "X360",
           2008
          ],
          [
           "Resonance of Fate",
           "PS3",
           2010
          ],
          [
           "Star Ocean: The Last Hope International",
           "PS3",
           2010
          ],
          [
           "The Last Remnant",
           "X360",
           2008
          ],
          [
           "Suikoden",
           "PS",
           1996
          ],
          [
           "Muramasa: The Demon Blade",
           "Wii",
           2009
          ],
          [
           "Lunar: Silver Star Story Complete",
           "PS",
           1999
          ],
          [
           "Tales of Symphonia: Dawn of the New World",
           "Wii",
           2008
          ],
          [
           "Folklore",
           "PS3",
           2007
          ],
          [
           "Shin Megami Tensei: Devil Survivor",
           "DS",
           2009
          ],
          [
           "Grandia",
           "PS",
           1999
          ],
          [
           "South Park: The Fractured But Whole",
           "NS",
           2018
          ],
          [
           "Valhalla Knights: Eldar Saga",
           "Wii",
           2009
          ],
          [
           "Ys VIII: Lacrimosa of Dana",
           "NS",
           2018
          ],
          [
           "Crimson Gem Saga",
           "PSP",
           2009
          ],
          [
           "Mount & Blade",
           "PC",
           2008
          ]
         ],
         "hovertemplate": "Genre=Role-Playing<br>Critic_Score=%{x}<br>Global_Sales=%{y}<br>Name=%{customdata[0]}<br>Platform=%{customdata[1]}<br>Year=%{customdata[2]}<extra></extra>",
         "legendgroup": "Role-Playing",
         "marker": {
          "color": "#FFA15A",
          "symbol": "circle"
         },
         "mode": "markers",
         "name": "Role-Playing",
         "orientation": "v",
         "showlegend": true,
         "type": "scatter",
         "x": [
          8,
          9,
          9.5,
          9.2,
          7.8,
          9.1,
          8.6,
          6.3,
          7.5,
          8.1,
          7.5,
          8.1,
          6.5,
          8.2,
          8,
          7.6,
          6.8,
          7.5,
          8.4,
          8.4,
          9.5,
          3.5,
          8.5,
          7.3,
          7.2
         ],
         "xaxis": "x",
         "y": [
          5.35,
          4.96,
          3.1,
          2.91,
          1.86,
          1.83,
          1.15,
          0.95,
          0.9,
          0.74,
          0.74,
          0.73,
          0.68,
          0.6,
          0.6,
          0.55,
          0.52,
          0.32,
          0.26,
          0.25,
          0.14,
          0.14,
          0.11,
          0.06,
          0.02
         ],
         "yaxis": "y"
        },
        {
         "customdata": [
          [
           "Street Fighter IV",
           "PS3",
           2009
          ],
          [
           "Street Fighter IV",
           "X360",
           2009
          ],
          [
           "Dragon Ball Z: Budokai Tenkaichi 3",
           "Wii",
           2007
          ],
          [
           "WWE Day of Reckoning 2",
           "GC",
           2005
          ],
          [
           "Castlevania Judgment",
           "Wii",
           2008
          ],
          [
           "BlazBlue: Calamity Trigger Portable",
           "PSP",
           2010
          ],
          [
           "Battle Fantasia",
           "X360",
           2008
          ]
         ],
         "hovertemplate": "Genre=Fighting<br>Critic_Score=%{x}<br>Global_Sales=%{y}<br>Name=%{customdata[0]}<br>Platform=%{customdata[1]}<br>Year=%{customdata[2]}<extra></extra>",
         "legendgroup": "Fighting",
         "marker": {
          "color": "#19d3f3",
          "symbol": "circle"
         },
         "mode": "markers",
         "name": "Fighting",
         "orientation": "v",
         "showlegend": true,
         "type": "scatter",
         "x": [
          9.3,
          9.3,
          7.5,
          8.3,
          4.6,
          7.9,
          7
         ],
         "xaxis": "x",
         "y": [
          4.19,
          2.95,
          1.03,
          0.34,
          0.16,
          0.11,
          0.09
         ],
         "yaxis": "y"
        },
        {
         "customdata": [
          [
           "Guitar Hero: On Tour",
           "DS",
           2008
          ],
          [
           "Guitar Hero III: Legends of Rock",
           "PS3",
           2007
          ],
          [
           "Penny-Punching Princess",
           "NS",
           2018
          ]
         ],
         "hovertemplate": "Genre=Misc<br>Critic_Score=%{x}<br>Global_Sales=%{y}<br>Name=%{customdata[0]}<br>Platform=%{customdata[1]}<br>Year=%{customdata[2]}<extra></extra>",
         "legendgroup": "Misc",
         "marker": {
          "color": "#FF6692",
          "symbol": "circle"
         },
         "mode": "markers",
         "name": "Misc",
         "orientation": "v",
         "showlegend": true,
         "type": "scatter",
         "x": [
          7.1,
          8.2,
          6.5
         ],
         "xaxis": "x",
         "y": [
          3.46,
          2.25,
          0.03
         ],
         "yaxis": "y"
        },
        {
         "customdata": [
          [
           "Heavy Rain",
           "PS3",
           2010
          ],
          [
           "Prince of Persia: The Sands of Time",
           "PS2",
           2003
          ],
          [
           "Grand Theft Auto: Chinatown Wars",
           "DS",
           2009
          ],
          [
           "Shadow of the Colossus",
           "PS2",
           2005
          ],
          [
           "Prince of Persia",
           "PS3",
           2008
          ],
          [
           "Prince of Persia",
           "X360",
           2008
          ],
          [
           "Silent Hill 3",
           "PS2",
           2003
          ],
          [
           "Okami",
           "PS2",
           2006
          ],
          [
           "Okami",
           "Wii",
           2008
          ],
          [
           "Hotel Dusk: Room 215",
           "DS",
           2007
          ],
          [
           "Silent Hill: Shattered Memories",
           "Wii",
           2009
          ],
          [
           "L.A. Noire",
           "NS",
           2017
          ],
          [
           "Afrika",
           "PS3",
           2009
          ],
          [
           "Geist",
           "GC",
           2005
          ]
         ],
         "hovertemplate": "Genre=Adventure<br>Critic_Score=%{x}<br>Global_Sales=%{y}<br>Name=%{customdata[0]}<br>Platform=%{customdata[1]}<br>Year=%{customdata[2]}<extra></extra>",
         "legendgroup": "Adventure",
         "marker": {
          "color": "#B6E880",
          "symbol": "circle"
         },
         "mode": "markers",
         "name": "Adventure",
         "orientation": "v",
         "showlegend": true,
         "type": "scatter",
         "x": [
          8.8,
          9,
          9.5,
          9.1,
          8.3,
          8.2,
          8.4,
          9.4,
          9,
          8,
          7.3,
          7.9,
          6,
          6.6
         ],
         "xaxis": "x",
         "y": [
          3.06,
          2.22,
          1.33,
          1.14,
          1.07,
          1.01,
          0.71,
          0.63,
          0.6,
          0.54,
          0.47,
          0.45,
          0.22,
          0.15
         ],
         "yaxis": "y"
        },
        {
         "customdata": [
          [
           "Halo Wars",
           "X360",
           2009
          ],
          [
           "New Play Control! Pikmin",
           "Wii",
           2009
          ],
          [
           "Fire Emblem: Radiant Dawn",
           "Wii",
           2007
          ],
          [
           "Sid Meier's Civilization Revolution",
           "DS",
           2008
          ],
          [
           "Little King's Story",
           "Wii",
           2009
          ],
          [
           "Sid Meier's Civilization II",
           "PC",
           1996
          ]
         ],
         "hovertemplate": "Genre=Strategy<br>Critic_Score=%{x}<br>Global_Sales=%{y}<br>Name=%{customdata[0]}<br>Platform=%{customdata[1]}<br>Year=%{customdata[2]}<extra></extra>",
         "legendgroup": "Strategy",
         "marker": {
          "color": "#FF97FF",
          "symbol": "circle"
         },
         "mode": "markers",
         "name": "Strategy",
         "orientation": "v",
         "showlegend": true,
         "type": "scatter",
         "x": [
          8.1,
          7.7,
          7.7,
          8.2,
          8.5,
          9.1
         ],
         "xaxis": "x",
         "y": [
          2.67,
          0.64,
          0.49,
          0.45,
          0.29,
          0
         ],
         "yaxis": "y"
        },
        {
         "customdata": [
          [
           "FIFA 18",
           "NS",
           2017
          ],
          [
           "NBA",
           "PSP",
           2005
          ],
          [
           "FIFA 07 Soccer",
           "GC",
           2006
          ]
         ],
         "hovertemplate": "Genre=Sports<br>Critic_Score=%{x}<br>Global_Sales=%{y}<br>Name=%{customdata[0]}<br>Platform=%{customdata[1]}<br>Year=%{customdata[2]}<extra></extra>",
         "legendgroup": "Sports",
         "marker": {
          "color": "#FECB52",
          "symbol": "circle"
         },
         "mode": "markers",
         "name": "Sports",
         "orientation": "v",
         "showlegend": true,
         "type": "scatter",
         "x": [
          5.8,
          6.4,
          8.3
         ],
         "xaxis": "x",
         "y": [
          1.1,
          0.21,
          0.18
         ],
         "yaxis": "y"
        },
        {
         "customdata": [
          [
           "Paper Mario: Color Splash",
           "WiiU",
           2016
          ],
          [
           "Starlink: Battle for Atlas",
           "NS",
           2018
          ],
          [
           "Monster Boy and the Cursed Kingdom",
           "NS",
           2018
          ]
         ],
         "hovertemplate": "Genre=Action-Adventure<br>Critic_Score=%{x}<br>Global_Sales=%{y}<br>Name=%{customdata[0]}<br>Platform=%{customdata[1]}<br>Year=%{customdata[2]}<extra></extra>",
         "legendgroup": "Action-Adventure",
         "marker": {
          "color": "#636efa",
          "symbol": "circle"
         },
         "mode": "markers",
         "name": "Action-Adventure",
         "orientation": "v",
         "showlegend": true,
         "type": "scatter",
         "x": [
          7.4,
          7.5,
          8
         ],
         "xaxis": "x",
         "y": [
          0.87,
          0.57,
          0.04
         ],
         "yaxis": "y"
        },
        {
         "customdata": [
          [
           "Snipperclips Plus: Cut It Out, Together!",
           "NS",
           2017
          ]
         ],
         "hovertemplate": "Genre=Puzzle<br>Critic_Score=%{x}<br>Global_Sales=%{y}<br>Name=%{customdata[0]}<br>Platform=%{customdata[1]}<br>Year=%{customdata[2]}<extra></extra>",
         "legendgroup": "Puzzle",
         "marker": {
          "color": "#EF553B",
          "symbol": "circle"
         },
         "mode": "markers",
         "name": "Puzzle",
         "orientation": "v",
         "showlegend": true,
         "type": "scatter",
         "x": [
          8.2
         ],
         "xaxis": "x",
         "y": [
          0.12
         ],
         "yaxis": "y"
        },
        {
         "customdata": [
          [
           "RollerCoaster Tycoon",
           "PC",
           1999
          ]
         ],
         "hovertemplate": "Genre=Simulation<br>Critic_Score=%{x}<br>Global_Sales=%{y}<br>Name=%{customdata[0]}<br>Platform=%{customdata[1]}<br>Year=%{customdata[2]}<extra></extra>",
         "legendgroup": "Simulation",
         "marker": {
          "color": "#00cc96",
          "symbol": "circle"
         },
         "mode": "markers",
         "name": "Simulation",
         "orientation": "v",
         "showlegend": true,
         "type": "scatter",
         "x": [
          8.7
         ],
         "xaxis": "x",
         "y": [
          0.04
         ],
         "yaxis": "y"
        }
       ],
       "layout": {
        "autosize": true,
        "legend": {
         "title": {
          "text": "Genre"
         },
         "tracegroupgap": 0
        },
        "margin": {
         "t": 60
        },
        "template": {
         "data": {
          "bar": [
           {
            "error_x": {
             "color": "#2a3f5f"
            },
            "error_y": {
             "color": "#2a3f5f"
            },
            "marker": {
             "line": {
              "color": "#E5ECF6",
              "width": 0.5
             },
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "bar"
           }
          ],
          "barpolar": [
           {
            "marker": {
             "line": {
              "color": "#E5ECF6",
              "width": 0.5
             },
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "barpolar"
           }
          ],
          "carpet": [
           {
            "aaxis": {
             "endlinecolor": "#2a3f5f",
             "gridcolor": "white",
             "linecolor": "white",
             "minorgridcolor": "white",
             "startlinecolor": "#2a3f5f"
            },
            "baxis": {
             "endlinecolor": "#2a3f5f",
             "gridcolor": "white",
             "linecolor": "white",
             "minorgridcolor": "white",
             "startlinecolor": "#2a3f5f"
            },
            "type": "carpet"
           }
          ],
          "choropleth": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "choropleth"
           }
          ],
          "contour": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "contour"
           }
          ],
          "contourcarpet": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "contourcarpet"
           }
          ],
          "heatmap": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "heatmap"
           }
          ],
          "heatmapgl": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "heatmapgl"
           }
          ],
          "histogram": [
           {
            "marker": {
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "histogram"
           }
          ],
          "histogram2d": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "histogram2d"
           }
          ],
          "histogram2dcontour": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "histogram2dcontour"
           }
          ],
          "mesh3d": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "mesh3d"
           }
          ],
          "parcoords": [
           {
            "line": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "parcoords"
           }
          ],
          "pie": [
           {
            "automargin": true,
            "type": "pie"
           }
          ],
          "scatter": [
           {
            "fillpattern": {
             "fillmode": "overlay",
             "size": 10,
             "solidity": 0.2
            },
            "type": "scatter"
           }
          ],
          "scatter3d": [
           {
            "line": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatter3d"
           }
          ],
          "scattercarpet": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattercarpet"
           }
          ],
          "scattergeo": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattergeo"
           }
          ],
          "scattergl": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattergl"
           }
          ],
          "scattermapbox": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattermapbox"
           }
          ],
          "scatterpolar": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterpolar"
           }
          ],
          "scatterpolargl": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterpolargl"
           }
          ],
          "scatterternary": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterternary"
           }
          ],
          "surface": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "surface"
           }
          ],
          "table": [
           {
            "cells": {
             "fill": {
              "color": "#EBF0F8"
             },
             "line": {
              "color": "white"
             }
            },
            "header": {
             "fill": {
              "color": "#C8D4E3"
             },
             "line": {
              "color": "white"
             }
            },
            "type": "table"
           }
          ]
         },
         "layout": {
          "annotationdefaults": {
           "arrowcolor": "#2a3f5f",
           "arrowhead": 0,
           "arrowwidth": 1
          },
          "autotypenumbers": "strict",
          "coloraxis": {
           "colorbar": {
            "outlinewidth": 0,
            "ticks": ""
           }
          },
          "colorscale": {
           "diverging": [
            [
             0,
             "#8e0152"
            ],
            [
             0.1,
             "#c51b7d"
            ],
            [
             0.2,
             "#de77ae"
            ],
            [
             0.3,
             "#f1b6da"
            ],
            [
             0.4,
             "#fde0ef"
            ],
            [
             0.5,
             "#f7f7f7"
            ],
            [
             0.6,
             "#e6f5d0"
            ],
            [
             0.7,
             "#b8e186"
            ],
            [
             0.8,
             "#7fbc41"
            ],
            [
             0.9,
             "#4d9221"
            ],
            [
             1,
             "#276419"
            ]
           ],
           "sequential": [
            [
             0,
             "#0d0887"
            ],
            [
             0.1111111111111111,
             "#46039f"
            ],
            [
             0.2222222222222222,
             "#7201a8"
            ],
            [
             0.3333333333333333,
             "#9c179e"
            ],
            [
             0.4444444444444444,
             "#bd3786"
            ],
            [
             0.5555555555555556,
             "#d8576b"
            ],
            [
             0.6666666666666666,
             "#ed7953"
            ],
            [
             0.7777777777777778,
             "#fb9f3a"
            ],
            [
             0.8888888888888888,
             "#fdca26"
            ],
            [
             1,
             "#f0f921"
            ]
           ],
           "sequentialminus": [
            [
             0,
             "#0d0887"
            ],
            [
             0.1111111111111111,
             "#46039f"
            ],
            [
             0.2222222222222222,
             "#7201a8"
            ],
            [
             0.3333333333333333,
             "#9c179e"
            ],
            [
             0.4444444444444444,
             "#bd3786"
            ],
            [
             0.5555555555555556,
             "#d8576b"
            ],
            [
             0.6666666666666666,
             "#ed7953"
            ],
            [
             0.7777777777777778,
             "#fb9f3a"
            ],
            [
             0.8888888888888888,
             "#fdca26"
            ],
            [
             1,
             "#f0f921"
            ]
           ]
          },
          "colorway": [
           "#636efa",
           "#EF553B",
           "#00cc96",
           "#ab63fa",
           "#FFA15A",
           "#19d3f3",
           "#FF6692",
           "#B6E880",
           "#FF97FF",
           "#FECB52"
          ],
          "font": {
           "color": "#2a3f5f"
          },
          "geo": {
           "bgcolor": "white",
           "lakecolor": "white",
           "landcolor": "#E5ECF6",
           "showlakes": true,
           "showland": true,
           "subunitcolor": "white"
          },
          "hoverlabel": {
           "align": "left"
          },
          "hovermode": "closest",
          "mapbox": {
           "style": "light"
          },
          "paper_bgcolor": "white",
          "plot_bgcolor": "#E5ECF6",
          "polar": {
           "angularaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "bgcolor": "#E5ECF6",
           "radialaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           }
          },
          "scene": {
           "xaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           },
           "yaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           },
           "zaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           }
          },
          "shapedefaults": {
           "line": {
            "color": "#2a3f5f"
           }
          },
          "ternary": {
           "aaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "baxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "bgcolor": "#E5ECF6",
           "caxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           }
          },
          "title": {
           "x": 0.05
          },
          "xaxis": {
           "automargin": true,
           "gridcolor": "white",
           "linecolor": "white",
           "ticks": "",
           "title": {
            "standoff": 15
           },
           "zerolinecolor": "white",
           "zerolinewidth": 2
          },
          "yaxis": {
           "automargin": true,
           "gridcolor": "white",
           "linecolor": "white",
           "ticks": "",
           "title": {
            "standoff": 15
           },
           "zerolinecolor": "white",
           "zerolinewidth": 2
          }
         }
        },
        "title": {
         "text": "Global Sales vs Critic Score by Genre"
        },
        "xaxis": {
         "anchor": "y",
         "autorange": true,
         "domain": [
          0,
          1
         ],
         "range": [
          3.102748865759274,
          10.397251134240726
         ],
         "title": {
          "text": "Critic_Score"
         },
         "type": "linear"
        },
        "yaxis": {
         "anchor": "x",
         "autorange": true,
         "domain": [
          0,
          1
         ],
         "range": [
          -1.0475984251968504,
          14.57759842519685
         ],
         "title": {
          "text": "Global_Sales"
         },
         "type": "linear"
        }
       }
      },
      "image/png": "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79si/LgVz9qndzyDC/sePXIw65mkaTjZYSVNAxfF7fAjHfw7m77xDf4by2M56RXH4JbDAc/Dj5YHx5fHztMrvvQ2/QlQNEn/GHIHJEACJEACJEACJEACJEACJEACJEACBhCgaGIAVJokARIgARIgARIgARIgARIgARIgARJIfwIUTdI/htwBCZAACZAACZAACZAACZAACZAACZCAAQQomhgAlSZJgARIgARIgARIgARIgARIgARIgATSnwBFk/SPIXdAAiRAAiRAAiRAAiRAAiRAAiRAAiRgAAGKJgZApUkSIAESIAESIAESIAESIAESIAESIIH0J0DRJP1jyB2QAAmQAAmQAAmQAAmQAAmQAAmQAAkYQICiiQFQaZIESIAESIAESIAESIAESIAESIAESCD9CVA0Sf8YcgckQAIkQAIkQAIkQAIkQAIkQAIkQAIGEKBoYgBUmiQBEiABEiABEiABEiABEiABEiABEkh/AhRN0j+G3AEJkAAJkAAJkAAJkAAJkAAJkAAJkIABBCiaGACVJkmABEiABEiABEiABEiABEiABEiABNKfAEWT9I8hd0ACJEACJEACJEACJEACJEACJEACJGAAAYomBkClSRIgARIgARIgARIgARIgARIgARIggfQnQNEk/WPIHZAACZAACZAACZAACZAACZAACZAACRhAgKKJAVBpkgRIgARIgARIgARIgARIgARIgARIIP0JUDRJ/xhyByRAAiRAAiRAAiRAAiRAAiRAAiRAAgYQoGhiAFSaJAESIAESIAESIAESIAESIAESIAESSH8CFE3SP4bcAQmQAAmQQIIIPD73NTz3r0Uhq1152VTcPOOSqB7UNTTh6jv+hEumTcaFU09F4P9vmXEJThg7omPegkWf4LWFSzHn4ZtQkJcTlx1t21WOGbf/EfsP1ITY+/sTd4SsHZfFaIQESIAESIAESIAEuiEBiibdMKjcEgmQAAmQQHwJBISOAWXFuO+2K5DhsPkXaHO6cc+jz+PiaZNjFiGiiSbx9Rj4as0mXH7jwwgXSMTry7/8tlOhJ96+0B4JkAAJkAAJkAAJpCsBiibpGjn6TQIkQAIkkDACIsOkorI2RDAJXzwghlzzy/Pxzgdf4N0lK3Du6eNx41UX4+Z7n4LILBk9YrBfZBHvBa6jRw72Z5d8/NlqrPh6Q8gaIvvk7kee7xh7uKyWYJ9i8TkwPjyD5v7br/BnxYgr+L0+JUWY+8gtGDKwzP+eEGD+OPc1/95mzp7nz2gJzA3wWLtxu3+sYBEsOCUseFyIBEiABEiABEiABHQQoGiiAx6nkgAJkAAJdH8C4cdrou04MK66tiFEWAjPLOnseE6waBJ+XEdktbzx7jL86NxJHZkundGP9bhPuLgi/Fu0ZAV+euGZfsEkWCwSIokQRwLCSSCbJVwQibRHLSJO97+ruEMSIAESIAESIIF0IUDRJF0iRT9JgARIgASSQiBQF2T2zOmdHsGJJoZ0RTRxulz+OijhdU+0AAjP9BBzw4/qiL3dOXseHpw5vSN7JLBGpP0EjiONP26UPxMlkGkSXodFCDY791SEHAHqbC0t++JYEiABEiABEiABEkgkAYomiaTNtUiABEiABNKOQDJEk/ID1Xj06Vcxe9b0uBSFDT/mEzhCI0SP+QuXRjw2I/YdyYdgQSSaaBKpYK4IfPjxnrS7GegwCZAACZAACZBAjyNA0aTHhZwbJgESIAES0EJA6/Gc8OyQrmSaxFs0Cd5v8LGdrTv2GSaaiDU76yqkJQYcSwIkQAIkQAIkQALJIkDRJFnkuS4JkAAJkEDaEOisHofIthDX0EF9Ix6pCRdNonXcEWJGoKZJPI7nLF66EqeOH6uqfxKcHVJb32TY8ZzworZpE2w6SgIkQAIkQAIkQAJBBCia8HYgARIgARIggcMQiNZyOHDsRdQK0SqalBYXhpZqq9wAAAVUSURBVGRiBIsmoqWxEGpWrtnk76xTkJfjb2+spRCsmL/ooy8jFqU9cewI/9oBAUdsP9DZRmshWNE9J7ymSeBI09Tvj+vYo1hrzotv4leXnhOXI0e8aUmABEiABEiABEggEQQomiSCMtcgARIgARLoFgTCa3UE1+iItRCsABEQFUSL3s5aDoevp6XlsFgnUm2R4HbCYkxAOAlug6y15XC4aCLsRipEq9X/bnHTcBMkQAIkQAIkQAJpTYCiSVqHj86TAAmQAAmQAAmQAAmQAAmQAAmQAAkYRYCiiVFkaZcESIAESIAEDCAQrTNN8FLhrYUNcIMmSYAESIAESIAESKBHEKBo0iPCzE2SAAmQAAmQAAmQAAmQAAmQAAmQAAloJUDRRCsxjicBEiABEiABEiABEiABEiABEiABEugRBCia9Igwc5MkQAIkQAIkQAIkQAIkQAIkQAIkQAJaCVA00UqM40mABEiABEiABEiABEiABEiABEiABHoEAYomPSLM3CQJkAAJkAAJkAAJkAAJkAAJkAAJkIBWAhRNtBLjeBIgARIgARIgARIgARIgARIgARIggR5BgKJJjwgzN0kCJEACJEACJEACJEACJEACJEACJKCVAEUTrcQ4ngRIgARIgARIgARIgARIgARIgARIoEcQoGjSI8LMTZIACZAACZAACZAACZAACZAACZAACWglQNFEKzGOJwESIAESIAESIAESIAESIAESIAES6BEEKJr0iDBzkyRAAiRAAiRAAiRAAiRAAiRAAiRAAloJUDTRSozjSYAESIAESIAESIAESIAESIAESIAEegQBiiY9IszcJAmQAAmQAAmQAAmQAAmQAAmQAAmQgFYCFE20EuN4EiABEiABEiABEiABEiABEiABEiCBHkGAokmPCDM3SQIkQAIkQAIkQAIkQAIkQAIkQAIkoJUARROtxDieBEiABEiABEiABEiABEiABEiABEigRxCgaNIjwsxNkgAJkAAJkAAJkAAJkAAJkAAJkAAJaCVA0UQrMY4nARIgARIgARIgARIgARIgARIgARLoEQQomvSIMHOTJEACJEACJEACJEACJEACJEACJEACWglQNNFKjONJgARIgARIgARIgARIgARIgARIgAR6BAGKJj0izNwkCZAACZAACZAACZAACZAACZAACZCAVgIUTbQS43gSIAESIAESIAESIAESIAESIAESIIEeQYCiSY8IMzdJAiRAAiRAAiRAAiRAAiRAAiRAAiSglQBFE63EOJ4ESIAESIAESIAESIAESIAESIAESKBHEKBo0iPCzE2SAAmQAAmQAAmQAAmQAAmQAAmQAAloJUDRRCsxjicBEiABEiABEiABEiABEiABEiABEugRBCia9Igwc5MkQAIkQAIkQAIkQAIkQAIkQAIkQAJaCVA00UqM40mABEiABEiABEiABEiABEiABEiABHoEAYomPSLM3CQJkAAJkAAJkAAJkAAJkAAJkAAJkIBWAhRNtBLjeBIgARIgARIgARIgARIgARIgARIggR5BgKJJjwgzN0kCJEACJEACJEACJEACJEACJEACJKCVAEUTrcQ4ngRIgARIgARIgARIgARIgARIgARIoEcQoGjSI8LMTZIACZAACZAACZAACZAACZAACZAACWglQNFEKzGOJwESIAESIAESIAESIAESIAESIAES6BEEKJr0iDBzkyRAAiRAAiRAAiRAAiRAAiRAAiRAAloJUDTRSozjSYAESIAESIAESIAESIAESIAESIAEegQBiiY9IszcJAmQAAmQAAmQAAmQAAmQAAmQAAmQgFYC/w/GoMMt2LsxRAAAAABJRU5ErkJggg==",
      "text/html": [
       "<div>                            <div id=\"187a8b1e-89f3-4119-9db3-7c221c2fa2ba\" class=\"plotly-graph-div\" style=\"height:525px; width:100%;\"></div>            <script type=\"text/javascript\">                require([\"plotly\"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById(\"187a8b1e-89f3-4119-9db3-7c221c2fa2ba\")) {                    Plotly.newPlot(                        \"187a8b1e-89f3-4119-9db3-7c221c2fa2ba\",                        [{\"customdata\":[[\"Call of Duty: Modern Warfare 2\",\"X360\",2009.0],[\"Call of Duty 4: Modern Warfare\",\"X360\",2007.0],[\"Call of Duty: World at War\",\"X360\",2008.0],[\"Call of Duty 4: Modern Warfare\",\"PS3\",2007.0],[\"Resistance: Fall of Man\",\"PS3\",2006.0],[\"Left 4 Dead\",\"X360\",2008.0],[\"Battlefield: Bad Company 2\",\"X360\",2010.0],[\"Killzone 2\",\"PS3\",2009.0],[\"Battlefield: Bad Company 2\",\"PS3\",2010.0],[\"BioShock\",\"X360\",2007.0],[\"Call of Duty 3\",\"X360\",2006.0],[\"Resistance 2\",\"PS3\",2008.0],[\"Call of Duty 3\",\"Wii\",2006.0],[\"Call of Duty 2\",\"X360\",2005.0],[\"Call of Duty: World at War\",\"Wii\",2008.0],[\"Crackdown\",\"X360\",2007.0],[\"Call of Duty: Modern Warfare: Reflex Edition\",\"Wii\",2009.0],[\"The House of the Dead 2 & 3 Return\",\"Wii\",2008.0],[\"BioShock\",\"PS3\",2008.0],[\"MAG: Massive Action Game\",\"PS3\",2010.0],[\"Army of Two\",\"PS3\",2008.0],[\"Star Fox: Assault\",\"GC\",2005.0],[\"Resistance: Retribution\",\"PSP\",2009.0],[\"Perfect Dark Zero\",\"X360\",2005.0],[\"Red Steel 2\",\"Wii\",2010.0],[\"Metroid Prime: Trilogy\",\"Wii\",2009.0],[\"Doom (2016)\",\"NS\",2017.0],[\"F.E.A.R. 2: Project Origin\",\"PC\",2009.0],[\"Wolfenstein\",\"PC\",2009.0],[\"Left 4 Dead\",\"PC\",2008.0]],\"hovertemplate\":\"Genre=Shooter<br>Critic_Score=%{x}<br>Global_Sales=%{y}<br>Name=%{customdata[0]}<br>Platform=%{customdata[1]}<br>Year=%{customdata[2]}<extra></extra>\",\"legendgroup\":\"Shooter\",\"marker\":{\"color\":\"#636efa\",\"symbol\":\"circle\"},\"mode\":\"markers\",\"name\":\"Shooter\",\"orientation\":\"v\",\"showlegend\":true,\"x\":[9.5,9.6,8.5,9.5,8.5,8.7,9.0,9.1,8.9,9.6,8.0,8.5,6.8,8.8,8.1,8.4,7.2,6.5,9.3,7.4,7.4,6.6,8.0,8.1,7.8,9.0,8.2,8.2,6.5,8.6],\"xaxis\":\"x\",\"y\":[13.53,9.41,7.5,6.72,4.37,3.52,3.48,3.02,2.96,2.83,2.7,2.47,2.24,2.06,1.94,1.75,1.51,1.45,1.44,1.32,1.17,1.08,0.9,0.77,0.62,0.61,0.43,0.09,0.04,0.02],\"yaxis\":\"y\",\"type\":\"scatter\"},{\"customdata\":[[\"Grand Theft Auto IV\",\"PS3\",2008.0],[\"Grand Theft Auto V\",\"XOne\",2014.0],[\"Uncharted 3: Drake's Deception\",\"PS3\",2011.0],[\"Uncharted 2: Among Thieves\",\"PS3\",2009.0],[\"Red Dead Redemption\",\"X360\",2010.0],[\"Metal Gear Solid 2: Sons of Liberty\",\"PS2\",2001.0],[\"Metal Gear Solid 4: Guns of the Patriots\",\"PS3\",2008.0],[\"Assassin's Creed\",\"X360\",2007.0],[\"Resident Evil 5\",\"PS3\",2009.0],[\"Uncharted: Drake's Fortune\",\"PS3\",2007.0],[\"Assassin's Creed\",\"PS3\",2007.0],[\"Monster Hunter: World\",\"PS4\",2018.0],[\"Tom Clancy's Splinter Cell\",\"XB\",2002.0],[\"inFAMOUS\",\"PS3\",2009.0],[\"Star Wars: The Force Unleashed\",\"Wii\",2008.0],[\"Devil May Cry 4\",\"PS3\",2008.0],[\"Tom Clancy's Splinter Cell: Pandora Tomorrow\",\"XB\",2004.0],[\"Resident Evil 4: Wii Edition\",\"Wii\",2007.0],[\"Bayonetta\",\"PS3\",2010.0],[\"Dante's Inferno\",\"PS3\",2010.0],[\"Bayonetta\",\"X360\",2010.0],[\"MadWorld\",\"Wii\",2009.0],[\"Tom Clancy's Splinter Cell: Double Agent\",\"X360\",2006.0],[\"Harry Potter and the Half-Blood Prince\",\"Wii\",2009.0],[\"X-Men Origins: Wolverine - Uncaged Edition\",\"PS3\",2009.0],[\"Tom Clancy's HAWX\",\"PS3\",2009.0],[\"No More Heroes\",\"Wii\",2008.0],[\"Fire Emblem Warriors\",\"NS\",2017.0],[\"Golden Axe: Beast Rider\",\"X360\",2008.0],[\"Dead Rising: Chop Till You Drop\",\"Wii\",2009.0],[\"Darksiders\",\"PS3\",2010.0],[\"Deadly Premonition\",\"X360\",2010.0],[\"Deadly Creatures\",\"Wii\",2009.0],[\"Dragon Ball: Revenge of King Piccolo\",\"Wii\",2009.0],[\"Disaster: Day of Crisis\",\"Wii\",2008.0],[\"Devil's Third\",\"WiiU\",2015.0]],\"hovertemplate\":\"Genre=Action<br>Critic_Score=%{x}<br>Global_Sales=%{y}<br>Name=%{customdata[0]}<br>Platform=%{customdata[1]}<br>Year=%{customdata[2]}<extra></extra>\",\"legendgroup\":\"Action\",\"marker\":{\"color\":\"#EF553B\",\"symbol\":\"circle\"},\"mode\":\"markers\",\"name\":\"Action\",\"orientation\":\"v\",\"showlegend\":true,\"x\":[10.0,9.0,9.3,9.5,9.5,9.5,9.3,8.2,8.6,8.7,8.2,9.3,9.3,8.7,6.8,8.2,9.2,9.2,8.6,7.3,8.9,8.1,8.5,6.0,7.0,7.5,8.2,7.3,4.0,6.7,7.8,6.0,6.9,6.1,6.4,3.8],\"xaxis\":\"x\",\"y\":[10.57,8.72,6.84,6.74,6.5,6.05,6.0,5.55,5.1,4.97,4.83,4.67,3.02,2.99,1.86,1.58,1.48,1.46,1.21,1.08,0.93,0.78,0.78,0.76,0.74,0.56,0.56,0.51,0.33,0.29,0.28,0.26,0.22,0.2,0.07,0.05],\"yaxis\":\"y\",\"type\":\"scatter\"},{\"customdata\":[[\"LittleBigPlanet\",\"PS3\",2008.0],[\"Jak and Daxter: The Precursor Legacy\",\"PS2\",2001.0],[\"Ratchet & Clank Future: A Crack in Time\",\"PS3\",2009.0],[\"Castlevania: Symphony of the Night\",\"PS\",1997.0],[\"Mirror's Edge\",\"PS3\",2008.0],[\"de Blob\",\"Wii\",2008.0],[\"NiGHTS: Journey of Dreams\",\"Wii\",2007.0],[\"Castlevania: Order of Ecclesia\",\"DS\",2008.0],[\"Ratchet & Clank Future: Quest for Booty\",\"PSN\",2008.0],[\"Wonder Boy: The Dragon's Trap (Remake)\",\"NS\",2017.0],[\"Runner3\",\"NS\",2018.0]],\"hovertemplate\":\"Genre=Platform<br>Critic_Score=%{x}<br>Global_Sales=%{y}<br>Name=%{customdata[0]}<br>Platform=%{customdata[1]}<br>Year=%{customdata[2]}<extra></extra>\",\"legendgroup\":\"Platform\",\"marker\":{\"color\":\"#00cc96\",\"symbol\":\"circle\"},\"mode\":\"markers\",\"name\":\"Platform\",\"orientation\":\"v\",\"showlegend\":true,\"x\":[9.4,9.0,8.7,9.3,7.7,8.1,7.0,8.2,7.7,7.4,7.5],\"xaxis\":\"x\",\"y\":[5.85,3.64,1.89,1.27,1.13,0.96,0.41,0.37,0.09,0.05,0.02],\"yaxis\":\"y\",\"type\":\"scatter\"},{\"customdata\":[[\"Forza Motorsport 3\",\"X360\",2009.0],[\"MotoGP '07\",\"X360\",2007.0]],\"hovertemplate\":\"Genre=Racing<br>Critic_Score=%{x}<br>Global_Sales=%{y}<br>Name=%{customdata[0]}<br>Platform=%{customdata[1]}<br>Year=%{customdata[2]}<extra></extra>\",\"legendgroup\":\"Racing\",\"marker\":{\"color\":\"#ab63fa\",\"symbol\":\"circle\"},\"mode\":\"markers\",\"name\":\"Racing\",\"orientation\":\"v\",\"showlegend\":true,\"x\":[9.2,7.7],\"xaxis\":\"x\",\"y\":[5.5,0.25],\"yaxis\":\"y\",\"type\":\"scatter\"},{\"customdata\":[[\"Final Fantasy XIII\",\"PS3\",2010.0],[\"Fallout 3\",\"X360\",2008.0],[\"Mass Effect 2\",\"X360\",2010.0],[\"Mass Effect\",\"X360\",2007.0],[\"The Legend of Dragoon\",\"PS\",2000.0],[\"Demon's Souls\",\"PS3\",2009.0],[\"The Elder Scrolls V: Skyrim\",\"NS\",2017.0],[\"White Knight Chronicles: International Edition\",\"PS3\",2010.0],[\"Lost Odyssey\",\"X360\",2008.0],[\"Tales of Vesperia\",\"X360\",2008.0],[\"Resonance of Fate\",\"PS3\",2010.0],[\"Star Ocean: The Last Hope International\",\"PS3\",2010.0],[\"The Last Remnant\",\"X360\",2008.0],[\"Suikoden\",\"PS\",1996.0],[\"Muramasa: The Demon Blade\",\"Wii\",2009.0],[\"Lunar: Silver Star Story Complete\",\"PS\",1999.0],[\"Tales of Symphonia: Dawn of the New World\",\"Wii\",2008.0],[\"Folklore\",\"PS3\",2007.0],[\"Shin Megami Tensei: Devil Survivor\",\"DS\",2009.0],[\"Grandia\",\"PS\",1999.0],[\"South Park: The Fractured But Whole\",\"NS\",2018.0],[\"Valhalla Knights: Eldar Saga\",\"Wii\",2009.0],[\"Ys VIII: Lacrimosa of Dana\",\"NS\",2018.0],[\"Crimson Gem Saga\",\"PSP\",2009.0],[\"Mount & Blade\",\"PC\",2008.0]],\"hovertemplate\":\"Genre=Role-Playing<br>Critic_Score=%{x}<br>Global_Sales=%{y}<br>Name=%{customdata[0]}<br>Platform=%{customdata[1]}<br>Year=%{customdata[2]}<extra></extra>\",\"legendgroup\":\"Role-Playing\",\"marker\":{\"color\":\"#FFA15A\",\"symbol\":\"circle\"},\"mode\":\"markers\",\"name\":\"Role-Playing\",\"orientation\":\"v\",\"showlegend\":true,\"x\":[8.0,9.0,9.5,9.2,7.8,9.1,8.6,6.3,7.5,8.1,7.5,8.1,6.5,8.2,8.0,7.6,6.8,7.5,8.4,8.4,9.5,3.5,8.5,7.3,7.2],\"xaxis\":\"x\",\"y\":[5.35,4.96,3.1,2.91,1.86,1.83,1.15,0.95,0.9,0.74,0.74,0.73,0.68,0.6,0.6,0.55,0.52,0.32,0.26,0.25,0.14,0.14,0.11,0.06,0.02],\"yaxis\":\"y\",\"type\":\"scatter\"},{\"customdata\":[[\"Street Fighter IV\",\"PS3\",2009.0],[\"Street Fighter IV\",\"X360\",2009.0],[\"Dragon Ball Z: Budokai Tenkaichi 3\",\"Wii\",2007.0],[\"WWE Day of Reckoning 2\",\"GC\",2005.0],[\"Castlevania Judgment\",\"Wii\",2008.0],[\"BlazBlue: Calamity Trigger Portable\",\"PSP\",2010.0],[\"Battle Fantasia\",\"X360\",2008.0]],\"hovertemplate\":\"Genre=Fighting<br>Critic_Score=%{x}<br>Global_Sales=%{y}<br>Name=%{customdata[0]}<br>Platform=%{customdata[1]}<br>Year=%{customdata[2]}<extra></extra>\",\"legendgroup\":\"Fighting\",\"marker\":{\"color\":\"#19d3f3\",\"symbol\":\"circle\"},\"mode\":\"markers\",\"name\":\"Fighting\",\"orientation\":\"v\",\"showlegend\":true,\"x\":[9.3,9.3,7.5,8.3,4.6,7.9,7.0],\"xaxis\":\"x\",\"y\":[4.19,2.95,1.03,0.34,0.16,0.11,0.09],\"yaxis\":\"y\",\"type\":\"scatter\"},{\"customdata\":[[\"Guitar Hero: On Tour\",\"DS\",2008.0],[\"Guitar Hero III: Legends of Rock\",\"PS3\",2007.0],[\"Penny-Punching Princess\",\"NS\",2018.0]],\"hovertemplate\":\"Genre=Misc<br>Critic_Score=%{x}<br>Global_Sales=%{y}<br>Name=%{customdata[0]}<br>Platform=%{customdata[1]}<br>Year=%{customdata[2]}<extra></extra>\",\"legendgroup\":\"Misc\",\"marker\":{\"color\":\"#FF6692\",\"symbol\":\"circle\"},\"mode\":\"markers\",\"name\":\"Misc\",\"orientation\":\"v\",\"showlegend\":true,\"x\":[7.1,8.2,6.5],\"xaxis\":\"x\",\"y\":[3.46,2.25,0.03],\"yaxis\":\"y\",\"type\":\"scatter\"},{\"customdata\":[[\"Heavy Rain\",\"PS3\",2010.0],[\"Prince of Persia: The Sands of Time\",\"PS2\",2003.0],[\"Grand Theft Auto: Chinatown Wars\",\"DS\",2009.0],[\"Shadow of the Colossus\",\"PS2\",2005.0],[\"Prince of Persia\",\"PS3\",2008.0],[\"Prince of Persia\",\"X360\",2008.0],[\"Silent Hill 3\",\"PS2\",2003.0],[\"Okami\",\"PS2\",2006.0],[\"Okami\",\"Wii\",2008.0],[\"Hotel Dusk: Room 215\",\"DS\",2007.0],[\"Silent Hill: Shattered Memories\",\"Wii\",2009.0],[\"L.A. Noire\",\"NS\",2017.0],[\"Afrika\",\"PS3\",2009.0],[\"Geist\",\"GC\",2005.0]],\"hovertemplate\":\"Genre=Adventure<br>Critic_Score=%{x}<br>Global_Sales=%{y}<br>Name=%{customdata[0]}<br>Platform=%{customdata[1]}<br>Year=%{customdata[2]}<extra></extra>\",\"legendgroup\":\"Adventure\",\"marker\":{\"color\":\"#B6E880\",\"symbol\":\"circle\"},\"mode\":\"markers\",\"name\":\"Adventure\",\"orientation\":\"v\",\"showlegend\":true,\"x\":[8.8,9.0,9.5,9.1,8.3,8.2,8.4,9.4,9.0,8.0,7.3,7.9,6.0,6.6],\"xaxis\":\"x\",\"y\":[3.06,2.22,1.33,1.14,1.07,1.01,0.71,0.63,0.6,0.54,0.47,0.45,0.22,0.15],\"yaxis\":\"y\",\"type\":\"scatter\"},{\"customdata\":[[\"Halo Wars\",\"X360\",2009.0],[\"New Play Control! Pikmin\",\"Wii\",2009.0],[\"Fire Emblem: Radiant Dawn\",\"Wii\",2007.0],[\"Sid Meier's Civilization Revolution\",\"DS\",2008.0],[\"Little King's Story\",\"Wii\",2009.0],[\"Sid Meier's Civilization II\",\"PC\",1996.0]],\"hovertemplate\":\"Genre=Strategy<br>Critic_Score=%{x}<br>Global_Sales=%{y}<br>Name=%{customdata[0]}<br>Platform=%{customdata[1]}<br>Year=%{customdata[2]}<extra></extra>\",\"legendgroup\":\"Strategy\",\"marker\":{\"color\":\"#FF97FF\",\"symbol\":\"circle\"},\"mode\":\"markers\",\"name\":\"Strategy\",\"orientation\":\"v\",\"showlegend\":true,\"x\":[8.1,7.7,7.7,8.2,8.5,9.1],\"xaxis\":\"x\",\"y\":[2.67,0.64,0.49,0.45,0.29,0.0],\"yaxis\":\"y\",\"type\":\"scatter\"},{\"customdata\":[[\"FIFA 18\",\"NS\",2017.0],[\"NBA\",\"PSP\",2005.0],[\"FIFA 07 Soccer\",\"GC\",2006.0]],\"hovertemplate\":\"Genre=Sports<br>Critic_Score=%{x}<br>Global_Sales=%{y}<br>Name=%{customdata[0]}<br>Platform=%{customdata[1]}<br>Year=%{customdata[2]}<extra></extra>\",\"legendgroup\":\"Sports\",\"marker\":{\"color\":\"#FECB52\",\"symbol\":\"circle\"},\"mode\":\"markers\",\"name\":\"Sports\",\"orientation\":\"v\",\"showlegend\":true,\"x\":[5.8,6.4,8.3],\"xaxis\":\"x\",\"y\":[1.1,0.21,0.18],\"yaxis\":\"y\",\"type\":\"scatter\"},{\"customdata\":[[\"Paper Mario: Color Splash\",\"WiiU\",2016.0],[\"Starlink: Battle for Atlas\",\"NS\",2018.0],[\"Monster Boy and the Cursed Kingdom\",\"NS\",2018.0]],\"hovertemplate\":\"Genre=Action-Adventure<br>Critic_Score=%{x}<br>Global_Sales=%{y}<br>Name=%{customdata[0]}<br>Platform=%{customdata[1]}<br>Year=%{customdata[2]}<extra></extra>\",\"legendgroup\":\"Action-Adventure\",\"marker\":{\"color\":\"#636efa\",\"symbol\":\"circle\"},\"mode\":\"markers\",\"name\":\"Action-Adventure\",\"orientation\":\"v\",\"showlegend\":true,\"x\":[7.4,7.5,8.0],\"xaxis\":\"x\",\"y\":[0.87,0.57,0.04],\"yaxis\":\"y\",\"type\":\"scatter\"},{\"customdata\":[[\"Snipperclips Plus: Cut It Out, Together!\",\"NS\",2017.0]],\"hovertemplate\":\"Genre=Puzzle<br>Critic_Score=%{x}<br>Global_Sales=%{y}<br>Name=%{customdata[0]}<br>Platform=%{customdata[1]}<br>Year=%{customdata[2]}<extra></extra>\",\"legendgroup\":\"Puzzle\",\"marker\":{\"color\":\"#EF553B\",\"symbol\":\"circle\"},\"mode\":\"markers\",\"name\":\"Puzzle\",\"orientation\":\"v\",\"showlegend\":true,\"x\":[8.2],\"xaxis\":\"x\",\"y\":[0.12],\"yaxis\":\"y\",\"type\":\"scatter\"},{\"customdata\":[[\"RollerCoaster Tycoon\",\"PC\",1999.0]],\"hovertemplate\":\"Genre=Simulation<br>Critic_Score=%{x}<br>Global_Sales=%{y}<br>Name=%{customdata[0]}<br>Platform=%{customdata[1]}<br>Year=%{customdata[2]}<extra></extra>\",\"legendgroup\":\"Simulation\",\"marker\":{\"color\":\"#00cc96\",\"symbol\":\"circle\"},\"mode\":\"markers\",\"name\":\"Simulation\",\"orientation\":\"v\",\"showlegend\":true,\"x\":[8.7],\"xaxis\":\"x\",\"y\":[0.04],\"yaxis\":\"y\",\"type\":\"scatter\"}],                        {\"template\":{\"data\":{\"histogram2dcontour\":[{\"type\":\"histogram2dcontour\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]}],\"choropleth\":[{\"type\":\"choropleth\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}],\"histogram2d\":[{\"type\":\"histogram2d\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]}],\"heatmap\":[{\"type\":\"heatmap\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]}],\"heatmapgl\":[{\"type\":\"heatmapgl\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]}],\"contourcarpet\":[{\"type\":\"contourcarpet\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}],\"contour\":[{\"type\":\"contour\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]}],\"surface\":[{\"type\":\"surface\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]}],\"mesh3d\":[{\"type\":\"mesh3d\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}],\"scatter\":[{\"fillpattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2},\"type\":\"scatter\"}],\"parcoords\":[{\"type\":\"parcoords\",\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"scatterpolargl\":[{\"type\":\"scatterpolargl\",\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"bar\":[{\"error_x\":{\"color\":\"#2a3f5f\"},\"error_y\":{\"color\":\"#2a3f5f\"},\"marker\":{\"line\":{\"color\":\"#E5ECF6\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"bar\"}],\"scattergeo\":[{\"type\":\"scattergeo\",\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"scatterpolar\":[{\"type\":\"scatterpolar\",\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"histogram\":[{\"marker\":{\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"histogram\"}],\"scattergl\":[{\"type\":\"scattergl\",\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"scatter3d\":[{\"type\":\"scatter3d\",\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"scattermapbox\":[{\"type\":\"scattermapbox\",\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"scatterternary\":[{\"type\":\"scatterternary\",\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"scattercarpet\":[{\"type\":\"scattercarpet\",\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"carpet\":[{\"aaxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"minorgridcolor\":\"white\",\"startlinecolor\":\"#2a3f5f\"},\"baxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"minorgridcolor\":\"white\",\"startlinecolor\":\"#2a3f5f\"},\"type\":\"carpet\"}],\"table\":[{\"cells\":{\"fill\":{\"color\":\"#EBF0F8\"},\"line\":{\"color\":\"white\"}},\"header\":{\"fill\":{\"color\":\"#C8D4E3\"},\"line\":{\"color\":\"white\"}},\"type\":\"table\"}],\"barpolar\":[{\"marker\":{\"line\":{\"color\":\"#E5ECF6\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"barpolar\"}],\"pie\":[{\"automargin\":true,\"type\":\"pie\"}]},\"layout\":{\"autotypenumbers\":\"strict\",\"colorway\":[\"#636efa\",\"#EF553B\",\"#00cc96\",\"#ab63fa\",\"#FFA15A\",\"#19d3f3\",\"#FF6692\",\"#B6E880\",\"#FF97FF\",\"#FECB52\"],\"font\":{\"color\":\"#2a3f5f\"},\"hovermode\":\"closest\",\"hoverlabel\":{\"align\":\"left\"},\"paper_bgcolor\":\"white\",\"plot_bgcolor\":\"#E5ECF6\",\"polar\":{\"bgcolor\":\"#E5ECF6\",\"angularaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"},\"radialaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"}},\"ternary\":{\"bgcolor\":\"#E5ECF6\",\"aaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"},\"baxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"},\"caxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"}},\"coloraxis\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"colorscale\":{\"sequential\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"sequentialminus\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"diverging\":[[0,\"#8e0152\"],[0.1,\"#c51b7d\"],[0.2,\"#de77ae\"],[0.3,\"#f1b6da\"],[0.4,\"#fde0ef\"],[0.5,\"#f7f7f7\"],[0.6,\"#e6f5d0\"],[0.7,\"#b8e186\"],[0.8,\"#7fbc41\"],[0.9,\"#4d9221\"],[1,\"#276419\"]]},\"xaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"white\",\"automargin\":true,\"zerolinewidth\":2},\"yaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"white\",\"automargin\":true,\"zerolinewidth\":2},\"scene\":{\"xaxis\":{\"backgroundcolor\":\"#E5ECF6\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"white\",\"gridwidth\":2},\"yaxis\":{\"backgroundcolor\":\"#E5ECF6\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"white\",\"gridwidth\":2},\"zaxis\":{\"backgroundcolor\":\"#E5ECF6\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"white\",\"gridwidth\":2}},\"shapedefaults\":{\"line\":{\"color\":\"#2a3f5f\"}},\"annotationdefaults\":{\"arrowcolor\":\"#2a3f5f\",\"arrowhead\":0,\"arrowwidth\":1},\"geo\":{\"bgcolor\":\"white\",\"landcolor\":\"#E5ECF6\",\"subunitcolor\":\"white\",\"showland\":true,\"showlakes\":true,\"lakecolor\":\"white\"},\"title\":{\"x\":0.05},\"mapbox\":{\"style\":\"light\"}}},\"xaxis\":{\"anchor\":\"y\",\"domain\":[0.0,1.0],\"title\":{\"text\":\"Critic_Score\"}},\"yaxis\":{\"anchor\":\"x\",\"domain\":[0.0,1.0],\"title\":{\"text\":\"Global_Sales\"}},\"legend\":{\"title\":{\"text\":\"Genre\"},\"tracegroupgap\":0},\"margin\":{\"t\":60},\"title\":{\"text\":\"Global Sales vs Critic Score by Genre\"}},                        {\"responsive\": true}                    ).then(function(){\n",
       "                            \n",
       "var gd = document.getElementById('187a8b1e-89f3-4119-9db3-7c221c2fa2ba');\n",
       "var x = new MutationObserver(function (mutations, observer) {{\n",
       "        var display = window.getComputedStyle(gd).display;\n",
       "        if (!display || display === 'none') {{\n",
       "            console.log([gd, 'removed!']);\n",
       "            Plotly.purge(gd);\n",
       "            observer.disconnect();\n",
       "        }}\n",
       "}});\n",
       "\n",
       "// Listen for the removal of the full notebook cells\n",
       "var notebookContainer = gd.closest('#notebook-container');\n",
       "if (notebookContainer) {{\n",
       "    x.observe(notebookContainer, {childList: true});\n",
       "}}\n",
       "\n",
       "// Listen for the clearing of the current output cell\n",
       "var outputEl = gd.closest('.output');\n",
       "if (outputEl) {{\n",
       "    x.observe(outputEl, {childList: true});\n",
       "}}\n",
       "\n",
       "                        })                };                });            </script>        </div>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# For this step, I would typically use libraries like Plotly or Bokeh.\n",
    "# Here I'm using Plotly:\n",
    "\n",
    "# Create an interactive scatter plot using Plotly\n",
    "import plotly.express as px\n",
    "fig = px.scatter(df_cleaned, x='Critic_Score', y='Global_Sales', color='Genre',\n",
    "                 hover_data=['Name', 'Platform', 'Year'])\n",
    "fig.update_layout(title='Global Sales vs Critic Score by Genre')\n",
    "fig.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ff89b1f3-e380-46c6-852d-873ad57c1f15",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
