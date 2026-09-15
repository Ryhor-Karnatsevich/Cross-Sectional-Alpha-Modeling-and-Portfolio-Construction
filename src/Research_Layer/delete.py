from research_storage import clear_current_research_outputs


def delete_research_layer_outputs():
    clear_current_research_outputs()
    print("Current Research Layer data and results deleted")


if __name__ == "__main__":
    delete_research_layer_outputs()
