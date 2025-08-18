// Enhanced functionality for haive-core documentation

document.addEventListener("DOMContentLoaded", function () {
  // Add smooth scrolling for anchor links
  document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
    anchor.addEventListener("click", function (e) {
      e.preventDefault();
      const target = document.querySelector(this.getAttribute("href"));
      if (target) {
        target.scrollIntoView({
          behavior: "smooth",
          block: "start",
        });
      }
    });
  });

  // Add reading progress indicator
  const progressBar = document.createElement("div");
  progressBar.className = "reading-progress";
  progressBar.innerHTML = '<div class="reading-progress-bar"></div>';
  document.body.appendChild(progressBar);

  const progressBarInner = progressBar.querySelector(".reading-progress-bar");

  window.addEventListener("scroll", () => {
    const windowHeight = window.innerHeight;
    const documentHeight = document.documentElement.scrollHeight - windowHeight;
    const scrolled = window.scrollY;
    const progress = (scrolled / documentHeight) * 100;
    progressBarInner.style.width = progress + "%";
  });

  // Enhance code blocks with language labels
  document.querySelectorAll(".highlight").forEach((block) => {
    const language = Array.from(block.classList).find((cls) =>
      cls.startsWith("language-"),
    );
    if (language) {
      const lang = language.replace("language-", "");
      const label = document.createElement("div");
      label.className = "code-language-label";
      label.textContent = lang;
      label.style.cssText = `
                position: absolute;
                top: 0;
                right: 0;
                background: var(--color-brand-primary);
                color: white;
                padding: 0.25rem 0.5rem;
                font-size: 0.75rem;
                border-radius: 0 0.5rem 0 0.25rem;
                font-family: var(--font-stack--monospace);
            `;
      block.style.position = "relative";
      block.appendChild(label);
    }
  });

  // Add collapsible sections for long API documentation
  document.querySelectorAll(".autoapi-nested-parse").forEach((section) => {
    if (section.scrollHeight > 500) {
      const wrapper = document.createElement("div");
      wrapper.className = "collapsible-section";
      wrapper.style.cssText =
        "position: relative; max-height: 500px; overflow: hidden;";

      const gradient = document.createElement("div");
      gradient.className = "gradient-overlay";
      gradient.style.cssText = `
                position: absolute;
                bottom: 0;
                left: 0;
                right: 0;
                height: 100px;
                background: linear-gradient(transparent, var(--color-background-primary));
                pointer-events: none;
            `;

      const button = document.createElement("button");
      button.textContent = "Show more";
      button.className = "expand-button";
      button.style.cssText = `
                position: absolute;
                bottom: 1rem;
                left: 50%;
                transform: translateX(-50%);
                background: var(--color-brand-primary);
                color: white;
                border: none;
                padding: 0.5rem 1rem;
                border-radius: 0.25rem;
                cursor: pointer;
                font-size: 0.875rem;
            `;

      section.parentNode.insertBefore(wrapper, section);
      wrapper.appendChild(section);
      wrapper.appendChild(gradient);
      wrapper.appendChild(button);

      button.addEventListener("click", function () {
        if (wrapper.style.maxHeight === "500px") {
          wrapper.style.maxHeight = "none";
          gradient.style.display = "none";
          button.textContent = "Show less";
          button.style.position = "static";
          button.style.marginTop = "1rem";
        } else {
          wrapper.style.maxHeight = "500px";
          gradient.style.display = "block";
          button.textContent = "Show more";
          button.style.position = "absolute";
          button.style.marginTop = "0";
        }
      });
    }
  });

  // Add search highlighting
  const urlParams = new URLSearchParams(window.location.search);
  const searchTerm = urlParams.get("highlight");
  if (searchTerm) {
    const content = document.querySelector(".bd-article");
    if (content) {
      highlightText(content, searchTerm);
    }
  }

  function highlightText(element, text) {
    const walker = document.createTreeWalker(
      element,
      NodeFilter.SHOW_TEXT,
      null,
      false,
    );

    const textNodes = [];
    let node;
    while ((node = walker.nextNode())) {
      textNodes.push(node);
    }

    textNodes.forEach((node) => {
      const content = node.textContent;
      const regex = new RegExp(`(${text})`, "gi");
      if (regex.test(content)) {
        const span = document.createElement("span");
        span.innerHTML = content.replace(regex, "<mark>$1</mark>");
        node.parentNode.replaceChild(span, node);
      }
    });
  }
});
