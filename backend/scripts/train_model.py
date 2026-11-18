import argparse


def main(argv=None):
    parser = argparse.ArgumentParser(description="Wrapper for the U-Net training entry point.")
    parser.add_argument("--register", default=None, help="Register an existing artifact bundle instead of training")
    parser.add_argument("--model-root", default=None)
    args, remaining = parser.parse_known_args(argv)

    if args.register:
        from app.core.config import get_settings
        from app.ml.model_registry import register_bundle

        settings = get_settings()
        dest = register_bundle(args.register, args.model_root or settings.model_root)
        print(f"Registered model bundle at {dest}")
        return
    from app.ml import train

    train.main(remaining)


if __name__ == "__main__":
    main()
