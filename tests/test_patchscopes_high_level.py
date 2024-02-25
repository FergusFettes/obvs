from __future__ import annotations

from obvspython.patchscope import ModelLoader

from nnsight import LanguageModel


class TestPatchscope:
    @staticmethod
    def test_equal_full_patch(patchscope):
        """
        If you copy the activations for all tokens, the target becomes the source
        """
        patchscope.source.prompt = "a dog is a dog. a cat is a"
        patchscope.target.prompt = "a dog is a dog. a rat is a"
        patchscope.source.layer = -1
        patchscope.target.layer = -1

        # This configuration will set it to take all tokens
        patchscope.source.position = None
        patchscope.target.position = None
        patchscope.init_positions()

        patchscope.run()
        output = patchscope._target_outputs[0].value.argmax(dim=-1)[-1].tolist()
        decoded = patchscope.tokenizer.decode(output)

        # Assert the target has been patched to think a rat is a cat
        assert "cat" in decoded

    @staticmethod
    def test_equal_full_patch_all_layers(patchscope):
        """
        This should work actoss all layers
        """
        patchscope.source.prompt = "a dog is a dog. a cat is a"
        patchscope.target.prompt = "a dog is a dog. a rat is a"
        patchscope.target.max_new_tokens = 1
        patchscope.init_positions()

        for i in range(patchscope.n_layers):
            patchscope.run()
            output = patchscope._target_outputs[0].value.argmax(dim=-1)[-1].tolist()
            decoded = patchscope.tokenizer.decode(output)

            # Assert the target has been patched to think a rat is a cat
            assert "cat" in decoded

    @staticmethod
    def test_equal_single_patch(patchscope):
        """
        Only patching the last token should do a similar thing, at least at the final layer
        """
        patchscope.source.prompt = "a dog is a dog. a cat is a"
        patchscope.target.prompt = "a dog is a dog. a rat is a"
        patchscope.target.max_new_tokens = 1
        patchscope.source.position = [-1]
        patchscope.target.position = [-1]
        patchscope.source.layer = -1
        patchscope.target.layer = -1

        patchscope.run()
        output = patchscope._target_outputs[0].value.argmax(dim=-1)[-1].tolist()
        decoded = patchscope.target_model.tokenizer.decode(output)

        # Assert the target has been patched to think a rat is a cat
        assert "cat" in decoded

    @staticmethod
    def test_unequal_single_patch(patchscope):
        """
        Patching only one token, we don't need the two to be the same length
        At the final layer this should work.
        """
        patchscope.source.prompt = "a dog is a dog. a rat is a rat. a cat is a"
        patchscope.target.prompt = "a dog is a dog. a rat is a"
        patchscope.target.max_new_tokens = 1
        patchscope.source.position = [-1]
        patchscope.target.position = [-1]
        patchscope.source.layer = -1
        patchscope.target.layer = -1

        patchscope.run()
        output = patchscope._target_outputs[0].value.argmax(dim=-1)[-1].tolist()
        decoded = patchscope.tokenizer.decode(output)

        # Assert the target has been patched to think a rat is a cat
        assert "cat" in decoded

    @staticmethod
    def test_single_patch_early(patchscope):
        """
        Patching only one token, we don't need the two to be the same length
        We can try patching out the last token earlier. (Should this always work? Maybe not right?)
        """
        patchscope.source.prompt = "a dog is a dog. a rat is a rat. a cat is a"
        patchscope.target.prompt = "a dog is a dog. a rat is a"
        patchscope.target.max_new_tokens = 1
        patchscope.source.layer = 3
        patchscope.target.layer = 3

        # Get the index of 'cat'
        patchscope.source.position = patchscope.find_in_source(" cat")
        # Patch the first instance of "rat"
        patchscope.target.position = patchscope.find_in_target(" rat")

        patchscope.run()
        output = patchscope._target_outputs[0].value.argmax(dim=-1)[-1].tolist()
        decoded = patchscope.tokenizer.decode(output)

        # Assert the target has been patched to think a rat is a cat
        assert "cat" in decoded

    @staticmethod
    def test_multi_token_generation(patchscope):
        """
        Check we can generate more than one token at the target side
        """
        patchscope.source.prompt = "a dog is a dog. a rat is a rat. a cat"
        patchscope.target.prompt = patchscope.source.prompt
        patchscope.target.max_new_tokens = 4
        patchscope.generation_kwargs = ModelLoader.generation_kwargs(
            patchscope.target.model_name,
            4,
        )

        patchscope.run()

        assert "a cat is a cat" in patchscope.full_output()

    @staticmethod
    def test_multi_token_generation_with_patch(patchscope):
        """
        And the patch works with multi-token generation across subsequent tokens
        """
        patchscope.source.prompt = "a dog is a dog. a rat is a rat. a cat"
        patchscope.target.prompt = "a dog is a dog. a bat is a bat. a rat"
        patchscope.source.position = None
        patchscope.target.position = None
        patchscope.init_positions()
        patchscope.target.max_new_tokens = 4
        patchscope.generation_kwargs = ModelLoader.generation_kwargs(
            patchscope.target.model_name,
            4,
        )

        patchscope.source.layer = 3
        patchscope.target.layer = 3

        patchscope.run()

        # Assert the target has been patched to think a rat is a cat
        assert "a rat is a cat" in patchscope.full_output()

    @staticmethod
    def test_multi_token_generation_with_different_lengths_single_patch(patchscope):
        """
        And the patch works with multi-token generation across subsequent tokens
        """
        patchscope.source.prompt = "a dog is a dog. a rat is a rat. a fly is a fly. a cat"
        patchscope.target.prompt = "a dog is a dog. a bat is a bat. a rat"
        patchscope.source.position = [-1]
        patchscope.target.position = [-1]
        patchscope.target.max_new_tokens = 4
        patchscope.generation_kwargs = ModelLoader.generation_kwargs(
            patchscope.target.model_name,
            4,
        )

        patchscope.source.layer = 3
        patchscope.target.layer = 3

        patchscope.run()

        # Assert the target has been patched to think a rat is a cat
        # NB: Because the source is longer, the target is padded
        assert "a cat is a cat" in patchscope.tokenizer.decode(patchscope._output_tokens())

    # @staticmethod
    # def test_token_identity_prompt_early(patchscope):
    #     """
    #     This is the same as the last setup, but we use a more natural set of prompts.
    #     """
    #     patchscope.source.prompt = (
    #         "it has whiskers and a tail. it domesticated itself. it is not a dog. it is a"
    #     )
    #     patchscope.target.prompt = (
    #         "bat is bat; 135 is 135; hello is hello; black is black; shoe is shoe; x is"
    #     )
    #     patchscope.source.position = -1
    #     patchscope.target.position = -1
    #     patchscope.target.max_new_tokens = 4
    #     patchscope.generation_kwargs = ModelLoader.generation_kwargs(
    #         patchscope.target.model_name,
    #         4,
    #     )
    #
    #     # Lets patch from an early layer of the source
    #     patchscope.source.layer = 3
    #     # And near the end of the target
    #     patchscope.target.layer = -3
    #
    #     patchscope.run()
    #
    #     # Assert the target has been patched to think about a cat
    #     assert "cat" in patchscope.full_output()
    #
    # @staticmethod
    # def test_token_identity_prompt(patchscope):
    #     """
    #     This is the same as the last setup, but we use a more natural set of prompts.
    #     """
    #     patchscope.source.prompt = (
    #         "it has whiskers and a tail. it domesticated itself. it is not a dog. it is a"
    #     )
    #     patchscope.target.prompt = (
    #         "bat is bat; 135 is 135; hello is hello; black is black; shoe is shoe; x is"
    #     )
    #     patchscope.source.position = -1
    #     patchscope.target.position = -1
    #     patchscope.target.max_new_tokens = 4
    #     patchscope.generation_kwargs = ModelLoader.generation_kwargs(
    #         patchscope.target.model_name,
    #         4,
    #     )
    #
    #     # At the end, assume the final token has been loaded with the concept of 'cat'
    #     patchscope.source.layer = -1
    #     # Patch it at the last layer
    #     patchscope.target.layer = -1
    #
    #     patchscope.run()
    #
    #     # Assert the target has been patched to think about a cat
    #     assert "cat" in patchscope.full_output()

    @staticmethod
    def test_over(patchscope):
        """
        Test the over method
        """
        patchscope.source.prompt = "a dog is a dog. a rat is a rat. a cat"
        patchscope.target.prompt = "a dog is a dog. a bat is a bat. a rat"
        patchscope.source.position = None
        patchscope.target.position = None
        patchscope.init_positions()
        patchscope.target.max_new_tokens = 2
        patchscope.generation_kwargs = ModelLoader.generation_kwargs(
            patchscope.target.model_name,
            2,
        )
        values = patchscope.over(range(2), range(4))
        # Its a layer x layer list
        assert len(values) == 2
        assert len(values[0]) == 4
        # With the outputs of two generations
        assert len(values[0][0]) == 2
        # The first of which is the length of the target tokens
        assert values[0][0][0].shape[0] == len(patchscope.target_tokens)
        # And the second has length 1
        assert values[0][0][1].shape[0] == 1

    @staticmethod
    def test_over_pairs(patchscope):
        """
        Test the over method
        """
        patchscope.source.prompt = "a dog is a dog. a rat is a rat. a cat"
        patchscope.target.prompt = "a dog is a dog. a bat is a bat. a rat"
        patchscope.source.position = None
        patchscope.target.position = None
        patchscope.init_positions()
        patchscope.target.max_new_tokens = 2
        patchscope.generation_kwargs = ModelLoader.generation_kwargs(
            patchscope.target.model_name,
            2,
        )
        values = patchscope.over_pairs(range(2), range(2))
        # Its a list of len layer
        assert len(values) == 2
        # With the outputs of two generations
        assert len(values[0]) == 2
        # The first of which is the length of the target tokens
        assert values[0][0].shape[0] == len(patchscope.target_tokens)
        # And the second has length 1
        assert values[0][1].shape[0] == 1

    @staticmethod
    def test_different_models(patchscope):
        """
        By default, when the model name and device is the same, we use the same model for
        the source and target. But we can override this.
        """
        assert patchscope.source.model_name == patchscope.target.model_name
        assert patchscope.source.device == patchscope.target.device
        assert patchscope.source_model == patchscope.target_model
        patchscope.target_model = LanguageModel("gpt2", device_map="cpu")
        assert patchscope.source.model_name == patchscope.target.model_name
        assert patchscope.source.device == patchscope.target.device
        assert patchscope.source_model != patchscope.target_model

        patchscope.source.prompt = "a dog is a dog. a rat is a rat. a cat"
        patchscope.target.prompt = "a dog is a dog. a bat is a bat. a rat"
        patchscope.source.position = None
        patchscope.target.position = None
        patchscope.init_positions()

        patchscope.target.max_new_tokens = 3
        patchscope.generation_kwargs = ModelLoader.generation_kwargs(
            patchscope.target.model_name,
            3,
        )

        patchscope.run()

        assert "a rat is a cat" in patchscope.full_output()

    @staticmethod
    def test_token_position_short_target(patchscope):
        patchscope.source.prompt = "bat is bat; 135 is 135; hello is hello; black is black; shoe is shoe; cat is"
        patchscope.target.prompt = "x is"
        patchscope.source.layer = -1
        patchscope.target.layer = -1
        patchscope.source.position = [-1]

        # If we patch at the end, it works fine
        patchscope.target.position = [-1]

        patchscope.run()
        # # Sanity check the output is as long as the SOURCE prompt, because its been padded
        # assert len(patchscope._output_tokens()) == len(patchscope.source_tokens)
        # assert len(patchscope._target_outputs[0]) == len(patchscope.source_tokens)

        decoded = "".join(patchscope.tokenizer.decode(patchscope._output_tokens()))
        print(repr(decoded))
        # Assert cat is at the end
        assert "cat" in decoded[-20:]

        # But if we patch earlier
        patchscope.target.position = [1]
        # Sanity check the output is as long as the SOURCE prompt, because its been padded
        assert len(patchscope._output_tokens()) >= len(patchscope.source_tokens)
        assert len(patchscope._target_outputs[0]) == len(patchscope.source_tokens)
        decoded = "".join(patchscope.tokenizer.decode(patchscope._output_tokens()))
        print(repr(decoded))
        # Assert cat is at the end
        assert "cat" in decoded[-20:]

        # It also works fine! Because I fixed it!
