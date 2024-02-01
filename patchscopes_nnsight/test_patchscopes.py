import pytest
from patchscopes_nnsight.patchscopes import SourceContext, TargetContext, Patchscope


# Make a patchscope fixture so we only have to load the model once. (This is slow)
@pytest.fixture(scope="session")
def patchscope():
    source_context = SourceContext(device="cpu")
    target_context = TargetContext.from_source(source_context, max_new_tokens=1)
    return Patchscope(source_context, target_context)


class TestPatchscope:
    @staticmethod
    def test_equal_full_patch(patchscope):
        patchscope.source.prompt = "a dog is a dog. a cat is a"
        patchscope.target.prompt = "a dog is a dog. a rat is a"
        patchscope.target.max_new_tokens = 1
        patchscope.get_position_and_layer()

        patchscope.run()
        output = patchscope._target_outputs[0].value.argmax(dim=-1)[-1].tolist()
        decoded = patchscope.target_model.tokenizer.decode(output)

        # Assert the target has been patched to think a rat is a cat
        assert "cat" in decoded

    @staticmethod
    def test_equal_single_patch(patchscope):
        patchscope.source.prompt = "a dog is a dog. a cat is a"
        patchscope.target.prompt = "a dog is a dog. a rat is a"
        patchscope.target.max_new_tokens = 1
        patchscope.source.position = -1
        patchscope.target.position = -1

        patchscope.run()
        output = patchscope._target_outputs[0].value.argmax(dim=-1)[-1].tolist()
        decoded = patchscope.target_model.tokenizer.decode(output)

        # Assert the target has been patched to think a rat is a cat
        assert "cat" in decoded

    @staticmethod
    def test_unequal_single_patch(patchscope):
        patchscope.source.prompt = "a dog is a dog. a rat is a rat. a cat is a"
        patchscope.target.prompt = "a dog is a dog. a rat is a"
        patchscope.target.max_new_tokens = 1
        patchscope.source.position = -1
        patchscope.target.position = -1

        patchscope.run()
        output = patchscope._target_outputs[0].value.argmax(dim=-1)[-1].tolist()
        decoded = patchscope.target_model.tokenizer.decode(output)

        # Assert the target has been patched to think a rat is a cat
        assert "cat" in decoded

    @staticmethod
    def test_multi_token_generation(patchscope):
        patchscope.source.prompt = "a dog is a dog. a rat is a rat. a cat"
        patchscope.target.prompt = "a dog is a dog. a rat is a rat. a cat"
        patchscope.target.max_new_tokens = 4

        patchscope.run()

        # Assert the target has been patched to think a rat is a cat
        assert "a cat is a cat" in "".join(patchscope.full_output())

    @staticmethod
    def test_multi_token_generation_with_patch(patchscope):
        patchscope.source.prompt = "a dog is a dog. a rat is a rat. a cat"
        patchscope.target.prompt = "a dog is a dog. a bat is a bat. a rat"
        patchscope.get_position_and_layer()
        patchscope.target.max_new_tokens = 4

        patchscope.run()

        # Assert the target has been patched to think a rat is a cat
        assert "a cat is a cat" in "".join(patchscope.full_output())
