from mealpy.utils.agent import Agent


class Layer:
    def __init__(
            self,
            layer_id: int,
            global_pop: list[Agent],
            local_pop: list[Agent],
            minimum_pop: int,
            n_epoch: int,
            ap: float,
            dyn_miu_cr: float,
            dyn_miu_f: float,
    ):
        self.dyn_local_pop: list[Agent] = []

        self.__layer_id = layer_id
        self.__global_pop = global_pop
        self.__local_pop = local_pop
        self.__minimum_pop = minimum_pop
        self.__n_epoch = n_epoch

        self.ap = ap
        self.dyn_miu_cr = dyn_miu_cr
        self.dyn_miu_f = dyn_miu_f

    @property
    def id(self):
        return self.__layer_id

    @property
    def global_pop(self):
        return self.__global_pop

    @property
    def local_pop(self):
        return self.__local_pop

    @property
    def minimum_pop(self):
        return self.__minimum_pop

    @property
    def n_epoch(self) -> int:
        return self.__n_epoch

    def evolve(self, *args, **kwargs):
        ...
