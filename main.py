{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv(r\"C:\\Users\\lx\\Desktop\\ML\\train.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Class_6    51811\n",
       "Class_8    51763\n",
       "Class_9    25542\n",
       "Class_2    24431\n",
       "Class_3    14798\n",
       "Class_7    14769\n",
       "Class_1     9118\n",
       "Class_4     4704\n",
       "Class_5     3064\n",
       "Name: target, dtype: int64"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "strs = df['target'].value_counts()\n",
    "strs"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'Class_1': 0,\n",
       " 'Class_2': 1,\n",
       " 'Class_3': 2,\n",
       " 'Class_4': 3,\n",
       " 'Class_5': 4,\n",
       " 'Class_6': 5,\n",
       " 'Class_7': 6,\n",
       " 'Class_8': 7,\n",
       " 'Class_9': 8}"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "value_map = dict((v, i) for i,v in enumerate(strs.index))\n",
    "value_map = {'Class_1': 0, 'Class_2': 1, 'Class_3': 2, 'Class_4': 3, 'Class_5': 4, 'Class_6': 5, 'Class_7': 6, 'Class_8': 7, 'Class_9': 8} \n",
    "value_map"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = df.replace({'target':value_map})\n",
    "df = df.drop(columns = ['id'])\n",
    "x_train = df.iloc[:, :-1]\n",
    "y_train = df['target']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv(r\"C:\\Users\\lx\\Desktop\\ML\\test.csv\")\n",
    "# df = df.drop(columns=['id'])\n",
    "x_test = df.iloc[:, 1:] # keep the id column for output"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.metrics import accuracy_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "model = RandomForestClassifier(n_jobs=2, n_estimators=500)\n",
    "model.fit(x_train, y_train)\n",
    "proba = model.predict_proba(x_test)\n",
    "# acc = accuracy_score(y_test, y_pred) * 100\n",
    "# print(\"\\nTesting Accuracy: {:.3f} %\".format(acc))\n",
    "output = pd.DataFrame({'id': df['id'], 'Class_1': proba[:,0], 'Class_2':proba[:,1], 'Class_3':proba[:,2], 'Class_4':proba[:,3], 'Class_5':proba[:,4], 'Class_6':proba[:,5], 'Class_7':proba[:,6], 'Class_8':proba[:,7], 'Class_9':proba[:,8]})\n",
    "output.to_csv(r'C:\\Users\\lx\\Desktop\\ML\\my_submission_rf.csv', index=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
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
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
