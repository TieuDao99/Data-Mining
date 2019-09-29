import pandas as pd
from pandas.api.types import is_numeric_dtype
import random as rd


def generate_datasets():
    names = ['abalone.csv', 'bank-additional.csv', 'metro_interstate_traffic_volume.csv']
    for i in range(len(names)):
        with open('Datasets\\' + names[i]) as f:
            d = pd.read_csv(f)

            new = d.head(1)
            for j in sorted(rd.sample(range(1, len(d)), 19)):
                new = new.append(d[j:j + 1])
            new.to_csv('Datasets\\'+names[i].split('.')[0]+'-20.csv', index=None)

            new2 = new.copy()
            for j in range(1, len(new2)):
                col = rd.choice(new2.columns)
                new2[col][j:j + 1] = "?"
            new2.to_csv('Datasets\\' + names[i].split('.')[0] + '-20-noise.csv', index=None)

def is_number(string):
    try:
        float(string)
    except ValueError:
        return False
    return True

def summary(d):
  def dtype(point):
    return 'numeric' if is_number(point) else 'nominal'

  with open('log.txt', mode='w') as f:
    f.write('# Number of instances: {}\n'.format(str(d.shape[0])))
    f.write('# Number of attributes: {}\n'.format(str(d.shape[1])))
    for i in range(d.shape[1]):
      f.write('# Attribute {}: {} {}\n'.format(str(i+1), d.columns[i], dtype(d[d.columns[i]][0])))

def replace(d):
    f = open('log.txt', mode='w')
    for i in range(d.shape[1]):
        if is_numeric_dtype(d[d.columns[i]]):
            continue
        else:
            if is_number(d[d.columns[i]][0]):
                temp = [float(d[d.columns[i]][0])]
                for j in range(1, d.shape[0]):
                    if is_number(d[d.columns[i]][j]):
                        temp.append(float(d[d.columns[i]][j]))
                for j in range(1, d.shape[0]):
                    if d[d.columns[i]][j] == '?':
                        d[d.columns[i]][j] = sum(temp)/len(temp)
                f.write('# Attribute: {}, {}, {} \n'.format(d.columns[i], d.shape[0]-len(temp), sum(temp)/len(temp)))
            else:
                count = 0
                for j in range(d.shape[0]):
                    if d[d.columns[i]][j] == '?':
                        count += 1
                        d[d.columns[i]][j] = d[d.columns[i]].value_counts().idxmax()
                if count == 0:
                    continue
                else:
                    f.write('# Attribute: {}, {}, {} \n'.format(d.columns[i], count, d[d.columns[i]].value_counts().idxmax()))
    f.close()
    d.to_csv('output.csv', index=False)

def discretize(d):
  no = int(input('number of bins: '))
  method = input('way of binning (width/depth): ')
  f = open('log.txt', mode='w')
  for i in range(d.shape[1]):
      if is_number(d[d.columns[i]][0]):
          f.write('# Attribute: {} \n'.format(d.columns[i]))
          temp = [float(d[d.columns[i]][0])]
          for j in range(1, d.shape[0]):
              if is_number(d[d.columns[i]][j]):
                  temp.append(float(d[d.columns[i]][j]))
          if method == 'width':
              interval = (max(temp) - min(temp))/len(temp)
              left = min(temp)
              right = left + interval
              for j in range(no):
                  count = 0
                  sumcol = 0
                  for k in range(d.shape[0]):
                      if is_number(d[d.columns[i]][k]):
                          if float(d[d.columns[i]][k]) >= left and float(d[d.columns[i]][k]) < right:
                              count += 1
                              sumcol += float(d[d.columns[i]][k])
                  for k in range(d.shape[0]):
                      if is_number(d[d.columns[i]][k]):
                          if float(d[d.columns[i]][k]) >= left and float(d[d.columns[i]][k]) < right:
                              d[d.columns[i]][k] = sumcol/count
                  f.write('[{}; {}]: {} \n'.format(left, right, count))
                  left = right
                  right += interval
          else:
              depth = int(len(temp)/no)
              temp.sort()
              bins = []
              for j in range(no):
                  one = []
                  if j == no - 1:
                      for k in range(len(temp)):
                          one.append(temp[k])
                  else:
                      for k in range(depth):
                          one.append(temp[k])
                  bins.append(one)
                  temp = temp[depth:]
              for j in range(no):
                  avg = sum(bins[j]) / len(bins[j])
                  for k in range(d.shape[0]):
                      if is_number(d[d.columns[i]][k]):
                          if float(d[d.columns[i]][k]) >= bins[j][0] and float(d[d.columns[i]][k]) <= bins[j][len(bins[j]) - 1]:
                              d[d.columns[i]][k] = avg
                  f.write('[{}; {}]: {} \n'.format(bins[j][0], bins[j][len(bins[j]) - 1], len(bins[j])))
      else:
          continue
  f.close()
  d.to_csv('output.csv', index=False)

def normalize(d):
  method = input('way of normalization (Min-max/Z-score): ')
  f = open('log.txt', mode='w')
  for i in range(d.shape[1]):
      if is_number(d[d.columns[i]][0]):
          f.write('# Attribute: {} [0.0; 1.0]\n'.format(d.columns[i]))
          temp = [float(d[d.columns[i]][0])]
          for j in range(1, d.shape[0]):
              if is_number(d[d.columns[i]][j]):
                  temp.append(float(d[d.columns[i]][j]))
          if method == 'Min-max':
              for j in range(d.shape[0]):
                  if is_number(d[d.columns[i]][j]):
                      d[d.columns[i]][j] = (float(d[d.columns[i]][j]) - min(temp))/(max(temp) - min(temp))
          else:
              temp = pd.DataFrame(temp)
              avg = d.mean(axis=0)[temp.columns[0]]
              std = d.std(axis=0)[temp.columns[0]]
              for j in range(d.shape[0]):
                  if is_number(d[d.columns[i]][j]):
                      d[d.columns[i]][j] = (float(d[d.columns[i]][j]) - avg)/std
      else:
          continue
  f.close()
  d.to_csv('output.csv', index=False)

def choose_option(opt, d):
    if opt == 'summary':
        summary(d)
    elif opt == 'replace':
        replace(d)
    elif opt == 'discretize':
        discretize(d)
    else:
        normalize(d)


if __name__ == "__main__":
    while(True):
        print('Wait a moment for regenerating datasets ... ')
        generate_datasets()
        print('Done!')
        print('Follow this syntax:\t\t <Labname> <Option> <Input file> <Output file> <Log file>\n'
              'With:'
              '\n\t\t- <Labname>     = {Lab1}'
              '\n\t\t- <Option>      = {summary; replace; discretize; normalize}'
              '\n\t\t- <Input file>  = {abalone.csv;'
              '\n\t\t                   abalone-20.csv;'
              '\n\t\t                   abalone-20-noise.csv;'
              '\n\t\t                   bank-additional.csv;'
              '\n\t\t                   bank-additional-20.csv;'
              '\n\t\t                   bank-additional-20-noise.csv;'
              '\n\t\t                   metro_interstate_traffic_volume.csv;'
              '\n\t\t                   metro_interstate_traffic_volume-20.csv;'
              '\n\t\t                   metro_interstate_traffic_volume-20-noise.csv}'
              '\n\t\t- <Output file> = {output.csv}'
              '\n\t\t- <Log file>    = {log.txt}')
        lab, option, infile, outfile, logfile = input().split()
        with open('Datasets\\'+infile) as f:
            data = pd.read_csv(f)
        choose_option(option, data)
        restart = input('Successfully compiled! Do you want to continue?(y/n)- ')
        if restart == 'y':
            continue
        else:
            break