from obvs.expansion import NucleusExpansion, Node


class TestNucleusExpansion:
    @staticmethod
    def test_expansion_init():
        expansion = NucleusExpansion("the quick brown fox jumps over the", cutoff_depth=0)
        assert len(expansion.nodes) == 1
        assert isinstance(expansion.nodes[0], Node)

        assert expansion.nodes[0].text == "the quick brown fox jumps over the"
        assert expansion.nodes[0].depth == 0
        assert expansion.nodes[0].parent is None
        assert expansion.nodes[0].probability == 1.0

    @staticmethod
    def test_expansion_depth(expansion):
        assert len(expansion.nodes) == 1
        expansion.cutoff_depth = 0
        expansion.expand()
        assert len(expansion.nodes) == 1

        expansion.reset()
        expansion.cutoff_depth = 1
        expansion.cutoff_prob = 1e-6
        expansion.cutoff_breadth = 10
        expansion.expand()
        assert len(expansion.nodes) == 11

        expansion.reset()
        expansion.cutoff_depth = 1
        expansion.cutoff_prob = 1e-6
        expansion.cutoff_breadth = 100
        expansion.expand()
        assert len(expansion.nodes) == 101

        expansion.reset()
        expansion.cutoff_depth = 1
        expansion.cutoff_prob = 0.5
        expansion.cutoff_breadth = 100
        expansion.expand()
        assert len(expansion.nodes) < 101

        expansion.reset()
        expansion.cutoff_depth = 2
        expansion.cutoff_prob = 0.01
        expansion.cutoff_breadth = 10
        expansion.expand()
        assert len(expansion.nodes) < 101
        assert expansion.nodes[0].get_children(expansion.nodes)[0].get_children(expansion.nodes)[0].depth == 2

    @staticmethod
    def test_validation_function(expansion):
        expansion.reset()
        expansion.cutoff_depth = 1
        expansion.cutoff_prob = 1e-6
        expansion.cutoff_breadth = 100

        def numbers_only(text: str) -> bool:
            return text.strip().isnumeric()

        expansion.validation_fn = numbers_only
        expansion.prompt = "1 2 3 4"
        expansion.expand()

        assert len(expansion.nodes) > 1
        assert all([node.text.isnumeric() for node in expansion.nodes[0].get_children(expansion.nodes)])
        # Get max probability
        max_prob = max([node.probability for node in expansion.nodes[0].get_children(expansion.nodes)])

        def no_numbers(text: str) -> bool:
            return not text.strip().isnumeric()

        expansion.validation_fn = no_numbers
        expansion.prompt = "1 2 3 4"
        expansion.expand()

        assert len(expansion.nodes) > 1
        assert all([not node.text.isnumeric() for node in expansion.nodes[0].get_children(expansion.nodes)])
        # Get max probability
        max_prob_no_numbers = max([node.probability for node in expansion.nodes[0].get_children(expansion.nodes)])

        assert max_prob > max_prob_no_numbers
