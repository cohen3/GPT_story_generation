from transformers import pipeline, set_seed


if __name__ == "__main__":
    generator = pipeline('text-generation', model='gpt2')
    set_seed(42)
    story = generator("the story begin as you find yourself in", max_length=30, num_return_sequences=5)
    for s in story:
        print(s)
# {'generated_text': "the story begin as you find yourself in a house full of children. In the house, you are told that you are the son of the God's"}
# {'generated_text': 'the story begin as you find yourself in a strange, outmoded mansion. To say it was dark is a understatement, and the only way out'}
# {'generated_text': "the story begin as you find yourself in the world of The Legend of Zelda: A Link to the Past. But you don't know where to start"}
# {'generated_text': 'the story begin as you find yourself in a series of rooms filled with small rooms with lights and lights with tiny lights in them, so you can see'}
# {'generated_text': 'the story begin as you find yourself in the middle of a fire – and quickly as you come face-to-face with a monster that you have'}


