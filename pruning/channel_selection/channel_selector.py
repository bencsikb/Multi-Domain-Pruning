from pruning.channel_selection.score_function import PuRLSelector


class ChannelSelector:
    def __init__(self, spec):

        if spec["ch_selector"] == "PuRL":
            return PuRLSelector(**spec)
        


