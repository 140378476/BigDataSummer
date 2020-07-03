import random

from project1.Core import Space2D, Agent, SpaceDrawer
from matplotlib import pyplot as plt

colors = ['blue', 'red', 'yellow', 'green']


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
        self.distribution = (7, 3)  # the distribution of types of agents, default: 70% to 30%,
        self.n = 5  # n neighbours
        self.k = 3  # removal when there are more than k neighbours of different type
        self.rules = []
        # a list of rules to decide whether to move, each item is a function that
        # takes this RemovalSimulation object and the agent as parameter

        self.selectionPercent = 0.7
        # choose an area where the percentage of the agents of the same type the above this
        self.maxSelectionCount = 5

        self.removalCounts = []  # count of removal of each tick

    def configure(self, percent, n, k):
        self.n = n
        self.k = k
        self.percent = percent

    def init(self):
        super().init()
        total = sum(self.distribution)
        for (i, d) in enumerate(self.distribution):
            n = int(d * self.agentCount / total)
            for _ in range(n):
                pos = self.randomPos()
                self.addAgent(RemovalAgent(i), pos)
        self.initRules()

    def tick(self):
        super().tick()
        counting = dict()
        for (i, a) in enumerate(self.agents):
            if self.decideRemoval(a):
                pos = self.selectRemovalPosition(a)
                self.moveAgent(a, pos)
                if a.type not in counting:
                    counting[a.type] = 1
                else:
                    counting[a.type] += 1

            # if i % 100 == 0:
            #     print(i)
        self.removalCounts.append(counting)
        print(counting)

    def decideRemoval(self, a):
        for r in self.rules:
            if r(a):  # one rule is satisfied
                return True
        return False

    def initRules(self):
        """
        Modify this method or directly add rules to `self.rules`.

        :return:
        """
        rules = self.rules

        rules.append(self.ruleType)
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

    def selectRemovalPosition(self, a):
        for _ in range(self.maxSelectionCount):
            pos = self.randomPos()
            blocks = self.adjacentBlocks(self.posToCord(pos), 2)
            totalCount = 0
            sameTypeCount = 0
            for cord in blocks:
                for t in self.getBlock(cord):
                    totalCount += 1
                    if a.type == t.type:
                        sameTypeCount += 1
            if totalCount == 0 or sameTypeCount / totalCount > self.selectionPercent:
                return pos

        return self.randomPos()


def saveStatistics(rs, folder="../out"):
    import os
    if not os.path.exists(folder):
        os.mkdir(folder)
    with open(f"{folder}/stat.txt", mode="w+") as f:
        f.write("Removal counts:\n")
        f.write("\n".join(map(lambda x: str(x), rs.removalCounts)))


if __name__ == '__main__':
    rs = RemovalSimulation(5000, 10000)  # 5000 * 5000, 10000 agents
    rs.distribution = (7, 2, 1)
    rs.init()
    drawer = SpaceDrawer(colorMap=RemovalAgent.getColor)
    # rs.addAgent(RemovalAgent(1), (2.5, 2.5))
    print("Counts of removal:")
    drawer.drawAndSave(rs)
    for i in range(50):
        rs.tick()
        if i < 20 or i % 5 == 0:
            drawer.drawAndSave(rs)
    saveStatistics(rs)
    # for a in rs.neighbours((2.5, 2.5), 6):
    #     print(a.pos)
    # plt.show()
