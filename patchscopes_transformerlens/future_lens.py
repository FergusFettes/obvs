from patchscopes_transformerlens.patchscopes import SourceContext, TargetContext, Patchscope

prompt = "John is a barber from Michigan. He likes hair and lakes."
source_context = SourceContext(
    prompt=prompt,
    device="cpu",
    layer=10
)

target_context = TargetContext.from_source(source_context)
target_context.prompt = "What is my name? _ _ _ _ _"

patchscope = Patchscope(source=source_context, target=target_context)

full_stop_token = patchscope.source_model.tokenizer.encode(" barber")[0]
prompt_tokens = patchscope.source_model.tokenizer.encode(prompt)
full_stop_index = prompt_tokens.index(full_stop_token)

underscore_token = patchscope.target_model.tokenizer.encode(" _")[0]
target_tokens = patchscope.target_model.tokenizer.encode(target_context.prompt)
underscore_index = target_tokens.index(underscore_token)

patchscope.source.position = full_stop_index
patchscope.target.position = underscore_index

patchscope.run()

for i in range(patchscope.target_model.cfg.n_layers):
    print(f"Layer {i}")
    patchscope.target.layer = i
    patchscope.run()
    print(patchscope.output())

patchscope.target_model.generate(target_context.prompt)




# THIS DIDNT WORK CAUSE YOU CANT GENERATE WITH HOOKS NICELY I think I'll go back to nnsight now.
