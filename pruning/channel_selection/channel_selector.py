from pruning.channel_selection.score_function import PuRLSelector


class ChannelSelector:
    def __new__(cls, spec):
        if spec.ch_selector == "PuRL":
            return PuRLSelector()
        else:
            raise ValueError(f"Unknown channel selector: {spec.ch_selector}")
