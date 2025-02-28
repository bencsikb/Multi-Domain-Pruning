class Codec:
    def __init__(self) -> None:
        pass

        self._encoded_range = [-1, 1]
        self._alpha_range = [0, 2.2]
        self._accuracy_range = [0, 1]
        self._params_range = [0, 64] # TODO
        self._channel_range = [0, 1024] # Should this be defined for each layer individually?

    def _calculate_dmap(self):
        pass
        # 

    def _calculate_spars(self):
        pass

    def _determine_pruned_area(self):
        """
        Not sure if needed, but get the pruned param from the data df and 
        leave the rest of the data for the layer if 1, delete if 0.
        """
        pass
        

    def encode(self):
        pass

    def decode(self):
        pass


# QUESTIONS:

# - mi jelezze a nem prunolást? Egy dedikált param vagy nem feltöltött state mátrix?
# - normálás minden layer channeljére külön-külön vagy a max channel méretre?
# - milyen normálási range legyen vagy milyen aktiváció, hogy az 1-nél nagyobb prop (dmap) is működjön?