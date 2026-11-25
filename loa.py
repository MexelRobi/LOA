"""Application entry point for the LOA interactive assistant."""

import argparse

from loa_storage import load_config


def main() -> int:
    """Load saved settings and start either the terminal or mobile interface."""
    parser = argparse.ArgumentParser(description="Local Operating Assistant")
    web_mode = parser.add_mutually_exclusive_group()
    web_mode.add_argument(
        "--web",
        action="store_true",
        help="start an authenticated mobile web chat on the local network",
    )
    web_mode.add_argument(
        "--install-web-service",
        action="store_true",
        help="install and start LOA web chat in the background at login",
    )
    web_mode.add_argument(
        "--uninstall-web-service",
        action="store_true",
        help="stop and remove the LOA web background service",
    )
    web_mode.add_argument(
        "--web-service-status",
        action="store_true",
        help="show whether the LOA web background service is running",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="web server bind address (default: all network interfaces)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="web server port (default: 8765)",
    )
    parser.add_argument(
        "--reset-web-password",
        action="store_true",
        help="replace the saved mobile web password before starting web mode",
    )
    parser.add_argument(
        "--certfile",
        help="TLS certificate file for HTTPS browser connections",
    )
    parser.add_argument(
        "--keyfile",
        help="TLS private key file paired with --certfile",
    )
    args = parser.parse_args()

    web_operation = args.web or args.install_web_service
    if args.reset_web_password and not args.web:
        parser.error("--reset-web-password requires --web")
    if bool(args.certfile) != bool(args.keyfile):
        parser.error("--certfile and --keyfile must be provided together")
    if (args.certfile or args.keyfile) and not web_operation:
        parser.error("--certfile and --keyfile require --web or --install-web-service")
    if (args.host != "0.0.0.0" or args.port != 8765) and not web_operation:
        parser.error("--host and --port require --web or --install-web-service")

    try:
        config = load_config()
    except RuntimeError as error:
        print(error)
        return 1

    if args.install_web_service:
        from loa_service import install_web_service

        try:
            plist_path = install_web_service(
                host=args.host,
                port=args.port,
                certfile=args.certfile,
                keyfile=args.keyfile,
            )
        except (OSError, RuntimeError, ValueError) as error:
            print(f"Could not install the LOA web background service: {error}")
            return 1
        print(f"LOA web background service installed and started: {plist_path}")
        print("It will start automatically when you sign in.")
        from loa_web import lan_ip_address

        service_host = args.host
        if service_host == "0.0.0.0":
            service_host = lan_ip_address() or "127.0.0.1"
        scheme = "https" if args.certfile else "http"
        print(f"Open {scheme}://{service_host}:{args.port}/ on a device on the same network.")
    elif args.uninstall_web_service:
        from loa_service import uninstall_web_service

        try:
            if uninstall_web_service():
                print("LOA web background service stopped and removed.")
            else:
                print("LOA web background service is not installed.")
        except (OSError, RuntimeError, ValueError) as error:
            print(f"Could not remove the LOA web background service: {error}")
            return 1
    elif args.web_service_status:
        from loa_service import web_service_status

        try:
            print(web_service_status())
        except (OSError, RuntimeError, ValueError) as error:
            print(f"Could not check the LOA web background service: {error}")
            return 1
    elif args.web:
        print("Loading web server dependencies...", flush=True)
        try:
            from loa_web import run_web_server

            run_web_server(
                config,
                host=args.host,
                port=args.port,
                reset_password=args.reset_web_password,
                certfile=args.certfile,
                keyfile=args.keyfile,
            )
        except KeyboardInterrupt:
            print("\nWeb chat stopped.")
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            print(f"Could not start the web server: {error}")
            return 1
    else:
        from loa_ui import LoaApp

        LoaApp(config).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
