import random

from project1.Core import Space2D, Agent
from matplotlib import pyplot as plt

colors = ['blue', 'red', 'green']


def typeToColor(type):
    if type < 0 or type >= len(colors):
        return 'black'
    return colors[type]


class RemovalAgent(Agent):

    def __init__(self, type) -> None:
        super().__init__()
        self.type = type

    def getColor(self):
        return typeToColor(self.type)


class RemovalSimulation(Space2D):

    def __init__(self, size, agentCount) -> None:
        super().__init__(size, agentCount)
        self.percent = 0.7  # 70% to 30%
        self.n = 5  # n neighbours
        self.k = 3  # removal when there are more than k neighbours of different type
        self.rules = []
        # a list of rules to decide whether to move, each item is a function that
        # takes this RemovalSimulation object and the agent as parameter

        self.removalCounts = []  # count of removal of each tick

    def configure(self, percent, n, k):
        self.n = n
        self.k = k
        self.percent = percent

    def init(self):
        super().init()
        count = int(self.agentCount * self.percent)
        for _ in range(count):
            pos = self.randomPos()
            self.addAgent(RemovalAgent(0), pos)
        for _ in range(count, self.agentCount):
            pos = self.randomPos()
            self.addAgent(RemovalAgent(1), pos)
        self.initRules()

    def tick(self):
        super().tick()
        count = 0
        for (i, a) in enumerate(self.agents):
            if self.decideRemoval(a):
                self.moveAgent(a, self.randomPos())
                count += 1
            # if i % 100 == 0:
            #     print(i)
        self.removalCounts.append(count)
        print(count)

    def decideRemoval(self, a):
        for r in self.rules:
            if r(a): # one rule is satisfied
                return True
        return False

    def initRules(self):
        """
        Modify this method or directly add rules to `self.rules`.

        :return:
        """
        rules = self.rules

        # rules.append(self.ruleType)
        rules.append(self.ruleRandom)

    def ruleType(self, a):
        """
        最近n个人，周围有k个及k个以上的人与本人肤色不同。

        :param a:
        :return:
        """
        neighbours = self.neighbours(a.pos, self.n + 1)
        count = 0
        for n in neighbours:
            if n.type != a.type:
                count += 1
        if count >= self.k:
            return True
        return False

    def ruleRandom(self, a):
        """
        在所有人中有2%的概率会随机搬家。

        :param a:
        :return:
        """
        return random.random() < 0.02


def drawAndSave(rs):
    fig = rs.draw(RemovalAgent.getColor)
    fig.savefig(f"../out/{rs.tickCount}.png", dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    rs = RemovalSimulation(5000, 10000)  # 5000 * 5000, 10000 agents
    rs.init()
    # rs.addAgent(RemovalAgent(1), (2.5, 2.5))
    drawAndSave(rs)
    for _ in range(30):
        rs.tick()
        drawAndSave(rs)
    # for a in rs.neighbours((2.5, 2.5), 6):
    #     print(a.pos)
    # plt.show()
