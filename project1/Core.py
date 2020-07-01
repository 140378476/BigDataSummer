import random
from typing import List, Tuple

import numpy as np
from matplotlib import pyplot as plt

class Agent:

    def __init__(self) -> None:
        super().__init__()
        self.space = None
        self.pos = None

    def init(self):
        pass


class Space2D:

    def __init__(self, size, agentCount) -> None:
        super().__init__()
        self.size = size
        self.agents = []
        self.agentCount = agentCount
        self._initBlocks(agentCount)
        self.tickCount = 0

    def _initBlocks(self, agentCount):
        blockCount = int(agentCount ** 0.5)
        blockSize = self.size / blockCount
        self.blocks: List[List[List[Agent]]] = [
            [[] for _ in range(blockCount)] for _ in range(blockCount)
        ]
        self.blockSize = blockSize
        self.blockCount = blockCount
        pass

    def init(self):
        """
        Override this method
        :return:
        """
        pass

    def containsBlock(self, cord):
        (x, y) = cord
        return 0 <= x < self.blockCount and 0 <= y < self.blockCount

    def contains(self, pos):
        (x, y) = pos
        return 0 <= x < self.size and 0 <= y < self.size

    def posToCord(self, pos):
        (x, y) = pos
        return int(x / self.blockSize), int(y / self.blockSize)

    def cordToPos(self, cord):
        (i, j) = cord
        return i * self.blockSize, j * self.blockSize

    def getBlock(self, cord):
        (i, j) = cord
        return self.blocks[i][j]

    def neighbourBlocks(self, cord, m=1):
        (i, j) = cord
        blocks = []
        for di in range(-m, m + 1):
            for dj in range(-m, m + 1):
                c = (i + di, j + dj)
                if not self.containsBlock(c):
                    continue
                blocks.append(c)
        return blocks

    def neighbourBlockBoundary(self, cord, n):
        (i, j) = cord
        blocks = []
        for d in range(-n, n + 1):
            cs = [(i + n, j + d),
                  (i - n, j + d),
                  (i + d, j + n),
                  (i + d, j - n),
                  ]
            for c in cs:
                if self.containsBlock(c):
                    blocks.append(c)
        return blocks

    def addAgent(self, agent: Agent, pos):
        agent.space = self
        self._putAgent(agent, pos)
        self.agents.append(agent)
        pass

    def _putAgent(self, agent: Agent, pos):
        cord = self.posToCord(pos)
        blockAgents = self.getBlock(cord)
        blockAgents.append(agent)
        agent.pos = pos

    def moveAgent(self, agent: Agent, pos):
        """
        Moves the given agent to a new position.

        """
        cord = self.posToCord(agent.pos)
        agents = self.getBlock(cord)
        agents.remove(agent)
        self._putAgent(agent, pos)

    def distanceSq(self, p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return dx * dx + dy * dy

    def minDistanceToGrid(self, pos):
        (x, y) = pos
        cord = self.posToCord(pos)
        (x0, y0) = self.cordToPos(cord)
        return min(x - x0, y - y0,
                   x0 + self.blockSize - x,
                   y0 + self.blockSize - y)

    def neighbours(self, pos, n):
        """
        Returns the nearest n agents in the space, sorted by distance. Note that
        the agent at the given position will also be returned.

        :param pos: the position
        :param n: the number of neighbours.
        :return:
        """
        cord = self.posToCord(pos)
        m = int(n ** 0.5 / 2)
        blocks = self.neighbourBlocks(cord, m)

        agents: List[Tuple[Agent, float]] = []

        def appendAgents(blocks, agents):
            for b in blocks:
                blockAgents = self.getBlock(b)
                for a in blockAgents:
                    d = self.distanceSq(pos, a.pos)
                    agents.append((a, d))

        appendAgents(blocks, agents)

        agents = sorted(agents, key=lambda pair: pair[1], reverse=True)
        threshold = self.minDistanceToGrid(pos) + m * self.blockSize
        d2 = threshold*threshold
        results = []
        while len(results) < n:
            if len(agents) > 0:
                (a, d) = agents[len(agents) - 1]
                if d <= d2:
                    results.append(a)
                    agents.pop()
                    continue
            m += 1
            threshold += self.blockSize
            d2 = threshold * threshold
            blocks = self.neighbourBlockBoundary(cord, m)
            if len(blocks) == 0:
                break
            appendAgents(blocks, agents)
            agents = sorted(agents, key=lambda pair: pair[1], reverse=True)
        while len(results) < n and len(agents) > 0:
            results.append(agents.pop()[0])
            continue
        return results

    def randomPos(self):
        x = random.random() * self.size
        y = random.random() * self.size
        return x, y

    def tick(self):
        """
        Override this method.
        :return:
        """
        self.tickCount += 1
        pass

    def draw(self, colorMap):
        fig, ax = plt.subplots()
        ax.set_xlim(0, self.size)
        ax.set_ylim(0, self.size)
        xs = []
        ys = []
        cs = []
        for a in self.agents:
            (x, y) = a.pos
            xs.append(x)
            ys.append(y)
            cs.append(colorMap(a))
        ax.scatter(xs, ys, color=cs,s=1.5)
        ax.set_title(f"{self.tickCount}-th iteration")
        return fig

    # def run(self):
    #     self.init()
