const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');

async function findTranscriptLinks(browser, pageNum = 1) {
  const url = 'https://seekingalpha.com/earnings/earnings-call-transcripts?page=' + pageNum;
  const page = await browser.newPage();
  await page.goto(url);
  let hrefs = await page.evaluate(() => {
    const anchors = document.querySelectorAll('a');
    return Array.from(anchors).map(anchor => anchor.href);
  });
  hrefs = hrefs.filter(href => href.match(/earnings\-call\-transcript$/));
  await page.close();
  return hrefs;
}

async function findTranscriptLinksMultiPage(browser, pages = 1) {
  let hrefs = [];
  for (let i = 1; i <= pages; i++) {
    console.log('Finding links on page ' + i);
    hrefs = hrefs.concat(await findTranscriptLinks(browser, i));
  }
  return hrefs;
}

async function scrapeTranscript(browser, url) {
  const page = await browser.newPage();
  await page.goto(url);

  const selector = '[data-test-id="article-content"]';

  await page.waitForSelector(selector);

  // find  > p innerText
  const transcript = await page.evaluate(() => {
    const parent = document.querySelector('[data-test-id="article-content"]');
    const paragraphs = parent.querySelectorAll('p');
    return Array.from(paragraphs).map(p => p.innerText);
  });

  await page.close();

  return transcript.join('\n');
}

async function main() {
  const browser = await puppeteer.launch({
    headless: true,
  });

  const LIVE = process.env.SCRAPE_LIVE === '1';
  const numPages = LIVE ? 1 : 100;
  const outputFolder = LIVE ? 'live' : 'output';

  const links = await findTranscriptLinksMultiPage(browser, numPages);
  console.log(links);
  for (let i = 0; i < links.length; i++) {
    const link = links[i];
    const filename = outputFolder + '/' + path.basename(link) + '.txt';
    if (!fs.existsSync(filename)) {
      try {
        const transcript = await scrapeTranscript(browser, link);
        fs.writeFileSync(filename, transcript);
      } catch (e) {
        console.log('Error scraping ' + link);
        console.log(e);
      }
    } else {
      console.log('Skipping ' + filename);
    }
  }
  await browser.close();
}

main().catch(console.error);
