from patchscopes_nnsight.patchscopes import SourceContext, TargetContext, Patchscope


# A source prompt with a salient entity
source_prompt = (
    "The Great Wall of China, a UNESCO World Heritage site. Built originally "
    "to protect the northern borders of the Chinese Empire from Xiongnu attacks "
    "during the reign of Emperor Qin Shi Huang, today it stands as a monument "
    "of Chinese civilization."
)

remote = True

# Setup source context with long passage containing a salient entity
source_context = SourceContext(
    prompt=source_prompt,  # Source prompt containing an entity
    model_name="gpt2-xl" if remote else "gpt2",
    position=-1,  # Last token position (assuming single input)
    layer=0,      # Start with the first layer
    device="cpu"
)

# Constructing a few-shot target prompt based on the source entity
target_context = TargetContext.from_source(source_context)
target_context.layer = -1
target_context.max_new_tokens = 20
target_context.prompt = (
    "Egyptian Pyramids: monumental structures of ancient Egypt, most of which are situated on the Giza Plateau; "
    "Roman Colosseum: a large amphitheatre in Rome, Italy; "
    "Taj Mahal: an ivory-white marble mausoleum on the right bank of the Yamuna river in the Indian city of Agra;"
)

# Now, setup the patchscope with the defined contexts
patchscope = Patchscope(source=source_context, target=target_context)
patchscope.REMOTE = remote
layers = len(patchscope.target_model.transformer.h)
step_size = layers // 10

target_tokens = patchscope.target_model.tokenizer.encode(target_context.prompt)
target_length = len(target_tokens)

# We will inspect the description generated by the model for the entity at different layers
entity_descriptions_by_layer = []
count = 0
for layer in range(1, layers - 1, step_size):
    print(f"Analyzing Layer {layer}. Iteration: {count} of {layers // step_size}")

    # Adjust the source context to the layer currently being inspected
    patchscope.source.layer = layer

    # Perform the patching operation and generate the description
    patchscope.run()

    # Retrieve and save the generated description for the entity
    generated_description = patchscope.full_output()
    full_joined = "".join(generated_description)
    print(f"Layer {layer} - Generated Description: {full_joined}")
    joined = "".join(generated_description[target_length - 1:])
    entity_descriptions_by_layer.append(joined)


joined = "\n".join(entity_descriptions_by_layer)
print(f"Entity Descriptions by Layer: {joined}")
