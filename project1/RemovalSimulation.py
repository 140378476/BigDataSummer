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
        self.n = 5
        self.percent = 0.7

    def configure(self, n, percent):
        self.n = n
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

    def tick(self):
        super().tick()
        count = 0
        for (i,a) in enumerate(self.agents):
            if self.decideRemoval(a):
                # print(a.pos)
                self.moveAgent(a, self.randomPos())
                count += 1
            # if i % 100 == 0:
            #     print(i)
        print(count)

    def decideRemoval(self, a):
        neighbours = self.neighbours(a.pos, self.n + 1)
        count = 0
        for n in neighbours:
            if n.type != a.type:
                count += 1
        return count >= 2

def drawAndSave(rs):
    fig = rs.draw(RemovalAgent.getColor)
    fig.savefig(f"../out/{rs.tickCount}.png")
    plt.close(fig)


if __name__ == '__main__':
    rs = RemovalSimulation(5000, 10000)
    rs.init()
    # rs.addAgent(RemovalAgent(1), (2.5, 2.5))
    drawAndSave(rs)
    for _ in range(20):
        rs.tick()
        drawAndSave(rs)
    # for a in rs.neighbours((2.5, 2.5), 6):
    #     print(a.pos)
    # plt.show()
