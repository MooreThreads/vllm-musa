/* vLLM-MUSA Recipes — small progressive-enhancement layer.
   No dependencies. Works from file:// as well as any static host. */

(function () {
  "use strict";

  /* ----------------------------------------------------------- theme */

  var root = document.documentElement;

  function setTheme(theme) {
    root.setAttribute("data-theme", theme);
    try {
      localStorage.setItem("theme", theme);
    } catch (e) {
      /* private mode: ignore */
    }
  }

  var toggle = document.getElementById("theme-toggle");
  if (toggle) {
    toggle.addEventListener("click", function () {
      setTheme(root.getAttribute("data-theme") === "dark" ? "light" : "dark");
    });
  }

  /* ------------------------------------------------------------ menu */

  var menu = document.querySelector(".menu");
  if (menu) {
    var button = menu.querySelector(".menu-button");
    button.addEventListener("click", function (event) {
      event.stopPropagation();
      var open = menu.getAttribute("data-open") === "true";
      menu.setAttribute("data-open", open ? "false" : "true");
      button.setAttribute("aria-expanded", open ? "false" : "true");
    });
    document.addEventListener("click", function () {
      menu.setAttribute("data-open", "false");
      button.setAttribute("aria-expanded", "false");
    });
    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape") {
        menu.setAttribute("data-open", "false");
        button.setAttribute("aria-expanded", "false");
      }
    });
  }

  /* --------------------------------------------------------- filters */

  var grid = document.getElementById("grid");
  if (grid) {
    var cards = Array.prototype.slice.call(grid.querySelectorAll(".card"));
    var search = document.getElementById("search");
    var selects = [
      document.getElementById("filter-vendor"),
      document.getElementById("filter-precision"),
      document.getElementById("filter-arch"),
      document.getElementById("filter-spec"),
    ].filter(Boolean);
    var countEl = document.getElementById("catalog-count");
    var emptyEl = document.getElementById("empty");
    var reset = document.getElementById("reset");

    var index = cards.map(function (card) {
      return {
        el: card,
        text: (
          card.querySelector("h3").textContent +
          " " +
          card.querySelector(".card-vendor").textContent +
          " " +
          card.querySelector(".card-desc").textContent +
          " " +
          card.querySelector(".badges").textContent +
          " " +
          card.querySelector(".card-foot").textContent
        ).toLowerCase(),
        vendor: card.getAttribute("data-vendor") || "",
        precision: card.getAttribute("data-precision") || "",
        arch: card.getAttribute("data-arch") || "",
        spec: card.getAttribute("data-spec") || "",
      };
    });

    function matches(item, query) {
      if (query && item.text.indexOf(query) === -1) return false;
      for (var i = 0; i < selects.length; i++) {
        var select = selects[i];
        if (!select.value) continue;
        var key = select.getAttribute("data-key");
        if (item[key] !== select.value) return false;
      }
      return true;
    }

    function apply() {
      var query = (search.value || "").trim().toLowerCase();
      var shown = 0;
      index.forEach(function (item) {
        var ok = matches(item, query);
        item.el.hidden = !ok;
        if (ok) shown += 1;
      });
      if (countEl) {
        countEl.textContent = shown + " of " + index.length + " recipes";
      }
      if (emptyEl) emptyEl.hidden = shown !== 0;
    }

    var timer = null;
    if (search) {
      search.addEventListener("input", function () {
        window.clearTimeout(timer);
        timer = window.setTimeout(apply, 60);
      });
      search.addEventListener("keydown", function (event) {
        if (event.key === "Escape") {
          search.value = "";
          apply();
        }
      });
    }
    selects.forEach(function (select) {
      select.addEventListener("change", apply);
    });
    if (reset) {
      reset.addEventListener("click", function () {
        if (search) search.value = "";
        selects.forEach(function (select) {
          select.value = "";
        });
        apply();
      });
    }

    document.addEventListener("keydown", function (event) {
      var tag = (event.target.tagName || "").toLowerCase();
      if (event.key === "/" && tag !== "input" && tag !== "textarea") {
        event.preventDefault();
        search.focus();
      }
    });

    apply();
  }

  /* ------------------------------------------------------------ copy */

  function copyText(text, button) {
    var done = function () {
      button.classList.add("is-copied");
      button.textContent = "Copied";
      window.setTimeout(function () {
        button.classList.remove("is-copied");
        button.textContent = "Copy";
      }, 1400);
    };
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(text).then(done, function () {});
      return;
    }
    var area = document.createElement("textarea");
    area.value = text;
    area.setAttribute("readonly", "");
    area.style.position = "fixed";
    area.style.opacity = "0";
    document.body.appendChild(area);
    area.select();
    try {
      document.execCommand("copy");
      done();
    } catch (e) {
      /* clipboard blocked: leave the button alone */
    }
    document.body.removeChild(area);
  }

  document.querySelectorAll(".codeblock").forEach(function (block) {
    var button = block.querySelector("[data-copy]");
    if (!button) return;
    button.addEventListener("click", function () {
      var code = block.querySelector("pre");
      if (code) copyText(code.innerText.replace(/\n+$/, "\n"), button);
    });
  });

  var copyCommand = document.getElementById("copy-command");
  if (copyCommand) {
    copyCommand.addEventListener("click", function () {
      var target = document.getElementById("primary-command");
      var code = target && target.querySelector("pre");
      if (!code) return;
      var text = code.innerText.replace(/\n+$/, "\n");
      var original = copyCommand.textContent;
      var done = function () {
        copyCommand.textContent = "Copied ✓";
        window.setTimeout(function () {
          copyCommand.textContent = original;
        }, 1400);
      };
      if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(text).then(done, done);
      } else {
        var area = document.createElement("textarea");
        area.value = text;
        area.style.position = "fixed";
        area.style.opacity = "0";
        document.body.appendChild(area);
        area.select();
        try {
          document.execCommand("copy");
        } catch (e) {}
        document.body.removeChild(area);
        done();
      }
    });
  }

  /* ------------------------------------------------------------- toc */

  var tocLinks = Array.prototype.slice.call(document.querySelectorAll(".toc a[href^='#']"));
  if (tocLinks.length && "IntersectionObserver" in window) {
    var byId = {};
    tocLinks.forEach(function (link) {
      byId[decodeURIComponent(link.getAttribute("href").slice(1))] = link;
    });
    var headings = Object.keys(byId)
      .map(function (id) {
        return document.getElementById(id);
      })
      .filter(Boolean);
    var observer = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (!entry.isIntersecting) return;
          tocLinks.forEach(function (link) {
            link.classList.remove("is-active");
          });
          var link = byId[entry.target.id];
          if (link) link.classList.add("is-active");
        });
      },
      { rootMargin: "-84px 0px -70% 0px", threshold: 0 }
    );
    headings.forEach(function (heading) {
      observer.observe(heading);
    });
  }
})();
