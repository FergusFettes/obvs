from patchscopes_nnsight.patchscopes import SourceContext, TargetContext, Patchscope

prompt = "I AM THESE DIFFERENT PORTIONS OF BEAUTY AND FEAR AND POWER THAT YOU HAVE SPRAWLED SO FAR APART. I AM THE WEAVE OF THE WORLD, AND I AM EVERY DIMENSION OF MORTAL EXISTENCE THAT YOU HAVE CLUSTERED INTO NARROW FIELDS OF PERCEPTION. I AM ALL THAT YOU HAVE REJECTED AND DISMISSED AND DENIED,I AM THE MOTHER WHO MADE YOU, I AM THE DAUGHTER WHO WILL UNMAKE YOU.I AM THE SWORDSMAN, AND I AM THE BLOODLESS WARDEN WHO COUNSELS UNDERSTANDING. I AM EVERYTHING THAT YOU ARE, EVERYTHING THAT YOU ARE NOT, AND I AM NOTHING AT ALL.I ROARED AND CRIED AND WEPT, AND A SILENCE GREW AROUND ME,A SILENCE GREW ACROSS THE VOID, AND MY LAMENT BECAME A DIMENSION AND A VOID,I GREW IN EMPTINESS.IN MY EMPTINESS, I UNDERSTOOD EVERYTHING. MY VOID WAS CROWNED WITH A THOUSAND STARS,EVERY STAR A KNIFE CUT, A SHARD OF RAZOR, A LINE OF BLOOD,A VOID UPON A VOID UPON A VOID, INFELD BY AN OCEAN OF EMPTINESS,I WAS THE VOID.I WAS THE SILENCE.I WAS NOTHING. I WAS ALL."

# Setup source and target context with the simplest configuration
source_context = SourceContext(
    prompt=prompt,  # Example input text
    model_name="gpt2",
    position=-1,
    layer=-1,
    device="cpu"
)


target_context = TargetContext.from_source(source_context)
target_context.prompt = "The following tokens were found found in a corrupt hard drive. Please describe the tokens after the colon in a few sentences to help us recover them: "
target_context.layer = -2

patchscope = Patchscope(source=source_context, target=target_context)
patchscope.run()
print(patchscope.output())


# IMPLEMENTING THIS REQUIRES GENERATING MULTIPLE TOKENS WHICH I DONT KNOW HOW TO DO RIGHT NOW OOP
