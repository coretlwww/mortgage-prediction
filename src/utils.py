import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

# Define a reusable function to plot multiple feature distributions in a 3x3 grid
def create_chart(df, cols, chart_title):
    fig, axes = plt.subplots(3, 3, figsize=(25, 12))  
    axes = axes.flatten()  # Flatten the 2D grid array into 1D for easy iteration

    for i, col in enumerate(cols):
        if i < len(axes): 
            # Call the plotting function (passed via chart_title argument) for the current column
            chart_title(data=df, x=col, ax=axes[i], color='lightgreen')
            
            # Rotate long category labels on the Job axis to prevent overlapping
            if col == "Job":
                axes[i].tick_params(axis="x", labelrotation=45)
                
            # Add text labels inside bars specifically for categorical columns
            if chart_title == sns.countplot:
                for container in axes[i].containers:
                    axes[i].bar_label(
                        container,
                        label_type="center",  
                        color="black",  
                        fontweight="bold",  
                        fontsize=10, 
                    )

    # Remove empty, unused subplots from the grid if features are fewer than 9
    for j in range(len(cols), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()  # Adjust spacing to avoid overlapping text and axes labels
    plt.show()


# Define a helper function to fit a pipeline and evaluate a model's Mean Absolute Error (MAE)
def get_mae(model, train_X, train_y, val_X, val_y, preprocessor):
    # Combine the preprocessor and the given model into a single executable pipeline
    new_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Train the complete pipeline on the training dataset
    new_pipeline.fit(train_X, train_y)
    
    # Generate predictions on the validation set and compute the absolute error metric
    preds_val = new_pipeline.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    
    return mae


# Define a reusable function to tune any model parameter and plot the resulting MAE curve
def plot_param_vs_mae(model_class, param_name, params_list, X_tr, y_tr, X_v, y_v, preprocessor, **extra_kwargs):
    # Enure the parameter values are unique and sorted from smallest to largest for a clean line plot
    sorted_params = sorted(list(set(params_list)))
    mae_scores = []
    
    # Loop through each parameter value to collect validation errors
    for value in sorted_params:
        model_kwargs = extra_kwargs.copy()
        model_kwargs[param_name] = value
        if "random_state" not in model_kwargs:
            model_kwargs["random_state"] = 0
            
        # Instantiate model, run your existing pipeline function, and save the score
        temp_model = model_class(**model_kwargs)
        mae = get_mae(temp_model, X_tr, y_tr, X_v, y_v, preprocessor)
        mae_scores.append(mae)
        print(f"Parameter {param_name} = {value:<4} | MAE: {mae:.4f}")

    # Set up and style the performance visualization chart
    plt.figure(figsize=(6, 5))
    sns.lineplot(x=sorted_params, y=mae_scores, marker='o', color='lightgreen', linewidth=2.5)
    
    # Dynamically label the chart based on the passed argument names
    model_name = model_class.__name__
    plt.title(f'{model_name}: Validation MAE vs. {param_name}', fontsize=14, fontweight='bold')
    plt.xlabel(f'{param_name}', fontsize=12)
    plt.ylabel('Mean Absolute Error (MAE)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.show()
    
    # Return the best value programmatically just like your previous function did
    best_mae_index = mae_scores.index(min(mae_scores))
    return sorted_params[best_mae_index]