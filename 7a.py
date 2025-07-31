def draw_gdg_logo():
    # Representing the GDG logo using lists
    left_shape = [("Blue", "Left"), ("Red", "Left")]
    right_shape = [("Green", "Right"), ("Yellow", "Right")]

    # Display symbolic representation
    print("GDG On Campus\n")
    
    print("Left Shape:")
    for color, position in left_shape:
        print(f"- {color} parallelogram on {position}")

    print("\nRight Shape:")
    for color, position in right_shape:
        print(f"- {color} parallelogram on {position}")

    print("\nREC")  # Bottom text

# Call the function to "draw" the logo using text
draw_gdg_logo()