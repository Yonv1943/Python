import multiprocessing
import time
from multiprocessing import Process, Pipe


class SubEnv(Process):
    def __init__(self, pipe):
        super().__init__()
        self.pipe = pipe

        state = 0  # state = env.reset()
        self.pipe.send(state)

    def run(self):
        for t in range(8):
            action = self.pipe.recv()
            state = action + 1
            time.sleep(1)
            self.pipe.send(state)


class VecEnv(Process):
    def __init__(self, pipes):
        super().__init__()
        self.pipes = pipes

    def run(self):
        states = [p.recv() for p in self.pipes]
        actions = states

        for i in range(8):
            [p.send(a) for p, a in zip(self.pipes, actions)]
            states = [p.recv() for p in self.pipes]
            actions = states
            print(';;;', states)


def run():
    print('Run Process')
    num_workers = 4

    pipes = [Pipe() for _ in range(num_workers)]
    pipe0s, pipe1s = list(map(list, zip(*pipes)))

    workers = [SubEnv(pipe0) for pipe0 in pipe0s]
    learner = VecEnv(pipe1s)

    process = [learner, ]
    process.extend(workers)

    [p.start() for p in process]
    [p.join() for p in process]
    [p.close() for p in process]


def main():
    print('Use Process to run Process')
    process = Process(target=run, args=())
    process.start()
    process.join()
    process.close()


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    run()  # run Process
    main()  # use Process to run Process
