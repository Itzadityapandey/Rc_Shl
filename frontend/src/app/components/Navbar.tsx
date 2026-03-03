"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";

export default function Navbar() {
    const pathname = usePathname();
    return (
        <nav className="navbar">
            <Link href="/" className="navbar-brand">
                <div className="brand-icon">⚡</div>
                SHL Recommender
            </Link>
            <ul className="navbar-links">
                <li>
                    <Link href="/" className={pathname === "/" ? "active" : ""}>
                        Recommend
                    </Link>
                </li>
                <li>
                    <Link href="/about" className={pathname === "/about" ? "active" : ""}>
                        About
                    </Link>
                </li>
            </ul>
        </nav>
    );
}
