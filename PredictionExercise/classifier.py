import math

class GNB(object):
    def __init__(self):
        self.possible_labels = ['left', 'keep', 'right']
        self.summaries = None

    def train(self, data, labels):
        """
        Trains the classifier with N data points and labels.

        INPUTS
        data - array of N observations
          - Each observation is a tuple with 4 values: s, d,
            s_dot and d_dot.
          - Example : [
                  [3.5, 0.1, 5.9, -0.02],
                  [8.0, -0.3, 3.0, 2.2],
                  ...
              ]

        labels - array of N labels
          - Each label is one of "left", "keep", or "right".
        """

        def mean(x):
            return sum(x)/float(len(x))

        def stdev(x):
            avg = mean(x)
            variance = sum([pow(x_-avg,2) for x_ in x])/float(len(x)-1)
            return math.sqrt(variance)

        def summarize(x):
            summaries = [(mean(x_), stdev(x_)) for x_ in zip(*x)]
            return summaries

        def separatedByClass(x, y):
            separated = {}
            for i in range(len(x)):
                if y[i] not in separated:
                    separated[y[i]] = []
                separated[y[i]].append(x[i])
            return separated

        def summarizeByClass(x, y):
            separated = separatedByClass(x, y)
            summaries = {}
            for i, v in separated.items():
                summaries[i] = summarize(v)
            return summaries

        self.summaries = summarizeByClass(data, labels)

        pass

    def predict(self, observation):
        """
        Once trained, this method is called and expected to return
        a predicted behavior for the given observation.

        INPUTS

        observation - a 4 tuple with s, d, s_dot, d_dot.
          - Example: [3.5, 0.1, 8.5, -0.2]

        OUTPUT

        A label representing the best guess of the classifier. Can
        be one of "left", "keep" or "right".
        """
        # TODO - complete this

        def calculateProbability(x, mean, stdev):
            exponent = math.exp(-(math.pow(x-mean,2))/(2*math.pow(stdev,2)))
            return (1/((math.sqrt(2*math.pi)*stdev))*exponent)

        def calculateClassProbability(summaries, inputVector):
            probabilities = {}
            for i, v in summaries.items():
                probabilities[i] = 1
                for k in range(len(v)):
                    mean, stdev = v[k]
                    x = inputVector[k]
                    probabilities[i] *= calculateProbability(x, mean, stdev)
            return probabilities

        probabilities = calculateClassProbability(self.summaries, observation)

        return max(probabilities, key=probabilities.get)
