## Replicate setup instructions step-by step:

1. Download docker desktop and launch it (if on mac)

2. Create the following structure:
```
        myapp/
        ├── Dockerfile
        ├── requirements.txt
        ├── your_python_script.py
```

3. Make sure you have all the dependencies installed in your environment and run: pip freeze > requirements.txt

4. On a Mac, set user variable using: ` export USER=$(id -u):$(id -g) `

5. The run:` docker build -t your-image-name . ` # Don't miss the dot at the end

6. Done. Now deploy this container anywhere and run: ` docker run --rm -it your-image-name `

7. We can also use Docker Hub to load the images remotely

8. Use: ` docker login `

9. Then to push:
```
        docker tag test_app:latest uonics/test_repo_uonics:latest
        docker push uonics/test_repo_uonics:latest
```

10. To pull: ` docker pull uonics/test_repo_uonics `

11. To run: ` docker run --rm -it uonics/test_repo_uonics `